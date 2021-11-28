import sys
import socket
import select
from SimPEG.inverse_problem import BaseInvProblem
import numpy as np

import gc
from SimPEG import (
    dias,
    maps,
    utils,
    data_misfit,
    regularization,
    optimization,
    inverse_problem,
    directives,
    inversion,
    objective_function,
    data
)
#from SimPEG import dask
from SimPEG import maps
from scipy.sparse import csr_matrix as csr
import os
from SimPEG.electromagnetics.static import resistivity as dc
import DCIPtools as DCIP
import discretize
from pymatsolver import Pardiso as solver
from pyMKL import mkl_set_num_threads
import json
import time
import struct

HOST = sys.argv[1] 
SOCKET_LIST = []
RECV_BUFFER = 160000
PORT = int(sys.argv[2])


def initialize(tile_config):
    """

        Method for initializing the tile for inversion. Creates a local
        data object and the local simulation to reduce computation time
        to global simulation

        input: configuration instruction for the local tile

    """

    # pull data from database and create local survey - (TODO make call from database for specific data and not use old python loader)
    patch = DCIP.loadDias(tile_config["dataset"])
    patch.readings = patch.readings[tile_config["sourcefrom"]:tile_config["sourceto"]]
    local_survey= patch.createDcSurvey("DC")

    # get electrode coverage of the local survey
    electrodes = np.vstack((local_survey.locations_a,
                            local_survey.locations_m,
                            local_survey.locations_n))

    min_east = np.min(electrodes[:, 0])
    min_north = np.min(electrodes[:, 1])
    max_east = np.max(electrodes[:, 0])
    max_north = np.max(electrodes[:, 1])

    # calculate the z components of x0
    global_active = np.asarray(tile_config["global_active"])
    z_max = np.asarray(tile_config["z_max"])
    # zblock = tile_config["z_cells"]
    tile_id = tile_config["tile_id"]

    # call utility that creates local/nested tensor meshes
    x_cells_global = [tuple(list_) for list_ in tile_config["x_cells_global"]]
    y_cells_global = [tuple(list_) for list_ in tile_config["y_cells_global"]]
    x_cells_local = [tuple(list_) for list_ in tile_config["x_cells_local"]]
    y_cells_local = [tuple(list_) for list_ in tile_config["y_cells_local"]]
    z_cells = [tuple(list_) for list_ in tile_config["z_cells"]]

    print("[SOME PARAMS] :", x_cells_global)

    # create base Mesh
    global_mesh = discretize.TensorMesh(
    [
        x_cells_global,
        y_cells_global,
        z_cells[::-1]
    ])

    # create 
    local_mesh = discretize.TensorMesh(
    [
        x_cells_local,
        y_cells_local,
        z_cells[::-1]
    ])

    # round to nearest 25
    total_cell_extent = np.sum(local_mesh.hz)
    base = 25
    z_max = base * round(z_max / 25)
    z0 = z_max - total_cell_extent

    # project mesh to proper utm coordinates
    global_mesh.x0 = tile_config["global_x0"]
    local_mesh.x0 = [min_east - np.sum(local_mesh.hx[:6]), min_north - np.sum(local_mesh.hy[:6]), z0]

    print('completed...')

    # create the local map
    local_map = maps.TileMap(global_mesh, global_active, local_mesh)

    # create the active map which excludes the air cells
    actmap = maps.InjectActiveCells(
        local_mesh, indActive=local_map.local_active, valInactive=np.log(1e-8)
    )

    # for DC we make an exponential mapping
    expmap = maps.ExpMap(local_mesh)
    mapping = expmap * actmap
    
    # Create the local simulation
    simulation = dc.Simulation3DNodal(
        local_mesh, survey=local_survey, sigmaMap=mapping,
        solver=solver)

    # simulation.mesh = TensorMesh([1])  # Light dummy
    del local_mesh,
    local_map.local_mesh = None
    actmap.mesh = None
    expmap.mesh = None

    simulation.sensitivity_path = './sensitivity/Tile' + str(tile_id) + '/'
    
    # set the data object
    data_object = data.Data(
        local_survey,
        dobs=local_survey.dobs,
        standard_deviation=local_survey.std,
    )
    # data_object.dobs = local_survey.dobs
    # data_object.standard_deviation = local_survey.std
    local_misfit = data_misfit.L2DataMisfit(
        data=data_object, simulation=simulation, model_map=local_map
    )
    local_misfit.W = 1 / local_survey.std

    print('completed...', local_survey.nD, local_misfit.simulation.mesh.nC, local_map.local_active.sum())

    return local_misfit


def getPredictedData(inputs, local_misfit):
    """
        Calculates the predicted data for local simulation

        inputs: 
        inputs = dictionary{'model vector':array, 'compute_j'=boolen}

        outputs:
        dpred = numpy array
    """

    time_dpred = time.time()

    # predicted data is requested upon every call to evaluate the objective function
    model = np.asarray(inputs["model"])

    # convert to local model size
    if local_misfit.model_map is not None:
        vec = local_misfit.model_map @ model
    else:
        vec = model

    dpred, f = local_misfit.simulation.dias_dpred(vec)

    # calculate the residuals
    residual = local_misfit.W * (local_misfit.data.dobs - dpred)
    
    print("time on machine: ", time.time() - time_dpred)
    return residual, f, model
    # return dpred


def getDeriv2(inputs, local_misfit, fields, model):
    """

        Calculates local contribution to J * vector result (implicit form)
        
    """

    v = np.asarray(inputs['vector'])

    # convert to local model size
    if local_misfit.model_map is not None:
        vec = local_misfit.model_map @ model
        v = local_misfit.model_map.deriv(model) @ v
    else:
        vec = model

    jvec = local_misfit.simulation.dias_Jvec(vec, v)
    w_jvec = local_misfit.W.diagonal() ** 2.0 * jvec
    jtwjvec = local_misfit.simulation.dias_Jtvec(vec, w_jvec)

    if getattr(local_misfit, "model_map", None) is not None:
        
        h_vec = csr.dot(jtwjvec, local_misfit.model_map.deriv(vec))

        return h_vec

    return jtwjvec


def getJtvec(v, local_misfit, fields, model):
    """

        Calculates local contribution to J^T * vector result (implicit form)

    """

    # convert to local model size
    if local_misfit.model_map is not None:

        vec = local_misfit.model_map @ model

    else:

        vec = model

    Jtvec = local_misfit.simulation.dias_Jtvec(vec, v, f=fields)

    if getattr(local_misfit, "model_map", None) is not None:

        h_vec = csr.dot(Jtvec, local_misfit.model_map.deriv(model))

        return h_vec

    return Jtvec


def recv_msg(sock):
    """
        Read message length and unpack it into an integer
    """

    raw_msglen = recvall(sock, 4)
    
    if not raw_msglen:
        return None
    
    msglen = struct.unpack('>I', raw_msglen)[0]
    
    # Read the message data
    return recvall(sock, msglen)


def recvall(sock, n):
    """
        Helper function to recv n bytes or return None if EOF is hit
    """
    data = bytearray()
    
    while len(data) < n:
        packet = sock.recv(n - len(data))
        
        if not packet:
            return None
        
        data.extend(packet)
    
    return data


def worker_server():
    """
        Worker node server code. The server will handle the decoupled mesh and its 
        accompanying simulation.

    """

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(10)
 
    # add server socket object to the list of readable connections
    SOCKET_LIST.append(server_socket)

    # global variables to be used for the sub simulations
    FIELDS = None
    LOCAL_MISFIT = None
    MODEL = None
    RESIDUAL = None
 
    print("v0.0.4 DIAS Worker server started on port " + str(PORT))
 
    # run loop in infinite loop
    while 1:

        # get the list sockets which are ready to be read through select
        ready_to_read,ready_to_write,in_error = select.select(SOCKET_LIST,[],[],0)

        for sock in ready_to_read:

            # a new connection request recieved
            if sock == server_socket: 
                sockfd, addr = server_socket.accept()
                SOCKET_LIST.append(sockfd)
                print("Client (%s, %s) connected to Adriatic worker" % addr)
                 
                broadcast(server_socket, sockfd, "[%s:%s] connected to worker\n" % addr)
             
            # a message from a client, not a new connection
            else:
                
                # process data recieved from client, 
                try:
                        # receiving data from the socket.
                    data = recv_msg(sock)
                    if data:

                        # convert data to a text stream and use json toconvert data to a usable format
                        request = str(data.decode('utf-8'))
                        split_request = request.split("!")

                        print("[REQUEST] :", len(request), len(split_request))
                        print(split_request[0])

                        # extract the streamed data
                        request_type = split_request[0]
                        params = json.loads(split_request[1])
                        
                        # create an output JSON
                        outputs = {}

                        # check request type and execute
                        if request_type == 'init':                            
                            # call initialization
                            print("\n\n here in init \n\n")
                            
                            LOCAL_MISFIT = initialize(params)

                            print("\n\n what now \n\n")
                            outputs["init"] = True

                            print("\n\n completed \n")

                        elif request_type == 'dpred':                            
                            # call dpred
                            print("\n\n here in dpred \n\n")
                            residual, FIELDS, MODEL = getPredictedData(params, LOCAL_MISFIT)
                            phi_d = 0.5 * np.vdot(residual, residual)
                            outputs["residual"] = phi_d
                            RESIDUAL = residual
                            print("\n\n completed residuals \n\n")

                        elif request_type == 'jtvec':                           
                            # call jtvec
                            print("\n\n here in jtvec \n\n")
                            wtw_d = LOCAL_MISFIT.W.diagonal() ** 2.0 * RESIDUAL
                            out_data = getJtvec(wtw_d, LOCAL_MISFIT, FIELDS, MODEL)
                            outputs["jtvec"] = out_data.tolist()
                            print("\n\n completed jtvec \n\n")
                        
                        elif request_type == 'derive2':                            
                            # call jvec
                            print("\n\n here in deriv2 \n\n")
                            out_data = getDeriv2(params, LOCAL_MISFIT, FIELDS, MODEL)
                            print("\n\n done deriv2 \n\n")
                            outputs["deriv2"] = out_data.tolist()
                            print("\n\n assigned deriv2 \n\n")

                        elif request_type == 'clean':                            
                            # call dpred
                            outputs["cleaned"] = clearLocalSimulation(params)

                        # there is something in the socket
                        message_out = json.dumps(outputs)
                        broadcast(server_socket, sock, message_out)  
                    else:
                        print("Here b/c it all done")
                        # remove the socket that's broken    
                        if sock in SOCKET_LIST:
                            SOCKET_LIST.remove(sock)

                        # at this stage, no data means probably the connection has been broken
                        broadcast(server_socket, sock, "Client (%s, %s) is offline2\n" % addr) 
                
                # exception 
                except:
                    print("service break!", sys.exc_info()[0])
                    broadcast(server_socket, sock, "Client (%s, %s) is offline\n" % addr)
                    continue

    server_socket.close()
    

def broadcast (server_socket, sock, message):
    """
        broadcast chat messages to all connected clients
    """

    for socket in SOCKET_LIST:
        # send the message only to peer
        if socket != server_socket:
            try :
                # construct final message that contains server instruction for size of data
                msg = struct.pack('>I', len(message)) + message.encode('utf-8')

                socket.sendall(msg)
            except :
                # broken socket connection
                socket.close()
                # broken socket, remove it
                if socket in SOCKET_LIST:
                    SOCKET_LIST.remove(socket)
 
if __name__ == "__main__":

    sys.exit(worker_server())        
