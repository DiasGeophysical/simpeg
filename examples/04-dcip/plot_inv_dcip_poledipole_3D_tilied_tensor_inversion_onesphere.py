
"""
Following example will show you how user can implement a 3D DC inversion.
"""

import discretize
import discretize.utils as meshutils
from discretize.tensor_mesh import TensorMesh
from SimPEG import dask
import dask
from SimPEG import (
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
from SimPEG.electromagnetics.static import resistivity as dc, utils as DCutils
from SimPEG.electromagnetics.static import induced_polarization as ip
from dask.distributed import Client, LocalCluster
from dask import config
from SimPEG.utils.drivers import create_tile_meshes, create_nested_mesh
import numpy as np
import matplotlib.pyplot as plt
from time import time

try:
    from pymatsolver.direct import Pardiso as solver
except ImportError:
    from SimPEG import SolverLU as solver

np.random.seed(12345)

def create_tensor_tile(survey, zblock, id, topo=None, min_cell_size=6.75, z_max=0):
    """
        A function for meshing a local tile for reduced resource
        demand for global mesh.
        
        NOTE: USED FOR NO REMOTE CONFIGURATIONS

        TODO: modify outputs to be instruction in order to use for cluster parallelism
        
        inputs: Survey = simpeg survey object
                z_blocks = vertical discretization
                id = tile id of the local simulation
                topo = DEM topography for surface discretization
                min_cell_size = size of smallest block
        
        outputs: discretize mesh object
    """

    # set the small cell zise
    da = min_cell_size

    # extract the electrode locations x,y,z excluding the remote
    electrodes_t1 = utils.uniqueRows(np.vstack((survey.locations_a,
                                survey.locations_m,
                                survey.locations_n)))
    electrodes_t1 = electrodes_t1[0]

    min_east = np.min(electrodes_t1[:, 0])
    min_north = np.min(electrodes_t1[:, 1])
    max_east = np.max(electrodes_t1[:, 0])
    max_north = np.max(electrodes_t1[:, 1])

    spread_t1 = ((max_east - min_east) / da, (max_north - min_north) / da)

    # put together the cells to cover to the remote and beyond
    x_cells = [] 
    x_cells += [(da, 5, -2), (da, np.ceil(spread_t1[0]))]
    x_cells += [(da, 5, 2)]
                            
    print(x_cells)

    # ===================================================================
    # now span enough cells to cover remote - Northing
    #

    # put together the cells to cover to the remote and beyond
    y_cells = [] 
    y_cells += [(da, 5, -2), (da, np.ceil(spread_t1[1]))]
    y_cells += [(da, 5, 2)]


    # ==============================================================
    # create the discretize object mesh
    #

    # create base Mesh
    tensor_mesh_file_output = "mesh-domino-tile" + str(id) + "-remote.msh"
    out_model_name = "SigmaOctree-pre_tile" + str(id) + ".dat"

    meshInput = discretize.TensorMesh(
    [
        x_cells,
        y_cells,
        zblock[::-1]
    ])

    # calculate the z components of x0
    total_cell_extent = np.sum(meshInput.hz)

    # round to nearest 25
    base = 25
    h_max = base * round(z_max / 25)
    z0 = h_max - total_cell_extent

    meshInput.x0 = [min_east - np.sum(meshInput.hx[:6]), min_north - np.sum(meshInput.hy[:6]), z0]

    print('[info] creating active cells')
    mesh = meshInput
    actinds = mesh.gridCC[:, 2] < 501
        
    model_tensor = np.ones(mesh.nC) * mesh.vol
    model_tensor[~actinds] = 1e-8    
    discretize.TensorMesh.writeUBC(mesh, tensor_mesh_file_output, models={out_model_name: model_tensor})

    print('completed...')

    return mesh


# @dask.delayed
def create_tile_dc(source, obs, uncert, global_mesh, global_active, tile_id, z_max):
    local_survey = dc.Survey(source)
    electrodes = np.vstack((local_survey.locations_a,
                            local_survey.locations_b,
                            local_survey.locations_m,
                            local_survey.locations_n))
    local_survey.dobs = obs
    local_survey.std = uncert
    # local_mesh = create_nested_mesh(
    #     electrodes, global_mesh, method="radial", max_distance=200.
    # )

    zblock = [ (25, 8), (30, 2), (40, 2), (50, 1), (75, 1), (100, 1), (200, 1), (200, 1)]

    local_mesh = create_tensor_tile(local_survey, zblock, tile_id, min_cell_size=25.0, z_max=z_max)

    local_map = maps.TileMap(global_mesh, global_active, local_mesh)

    actmap = maps.InjectActiveCells(
        local_mesh, indActive=local_map.local_active, valInactive=np.log(1e-8)
    )

    expmap = maps.ExpMap(local_mesh)
    mapping = expmap * actmap
    # Create the local misfit
    max_chunk_size = 256
    simulation = dc.Simulation3DNodal(
        local_mesh, survey=local_survey, sigmaMap=mapping, storeJ=True,
        solver=solver, max_ram=1
    )

    simulation.mesh = TensorMesh([1])  # Light dummy
    del local_mesh,
    local_map.local_mesh = None
    actmap.mesh = None
    expmap.mesh = None

    simulation.sensitivity_path = './sensitivity/Tile' + str(tile_id) + '/'
    data_object = data.Data(
        local_survey,
        dobs=obs,
        standard_deviation=uncert,
    )
    data_object.dobs = obs
    data_object.standard_deviation = uncert
    local_misfit = data_misfit.L2DataMisfit(
        data=data_object, simulation=simulation, model_map=local_map
    )
    local_misfit.W = 1 / uncert

    return local_misfit


#####################################
# creating survey design helper class
# -----------------------------------
class transmit():
    def __init__(self):
        self.receivers = []
        self.transmit = []
        self.dipoles = []

    def createDipoles(self, dp_min, dp_max):
        # print('len dp: ', self.receivers.shape)
        for ii in range(self.receivers.shape[1]):
            node1 = self.receivers[:, ii]
            for jj in range(ii, self.receivers.shape[1]):
                node2 = self.receivers[:, jj]
                dist = np.sum(np.asarray([node1 - node2])**2)**0.5 
                if dp_min < dist < dp_max:
                    self.dipoles.append([node1, node2])

def run(survey_type="pole-dipole", plotIt=True):

    # # create our dask worker managers
    cluster = LocalCluster(processes=False)
    client = Client(cluster)

    max_chunk_size = 256

    config.set({"array.chunk-size": str(max_chunk_size) + "MiB"})

    ##############################################################################
    # create rolling type survey
    # --------------------------
    transmits = []
    length = 1000
    theta = np.pi / 2
    num_stations = 10
    num_stations_extend_tx = 3
    tx_rx_offset = 50
    num_lines = 10
    line_spacing = 100
    tx_site_spacing = 100
    num_active_lead = 2
    num_active_behind = 2

    # get end points
    x0, y0 = 0., 0.
    x1, y1 = 0., length
    # get number of station (base)
    number_of_stns = num_stations
    station_spacing = length / number_of_stns
    stns_y = np.arange(y0, y1, station_spacing)
    stns_x = np.zeros(stns_y.shape[0])
    tx_end_length = station_spacing * num_stations_extend_tx
    # get end points tx
    x0_tx = x0
    if tx_rx_offset == 0:
        y0_tx = y0 - tx_end_length
    else:
        y0_tx = y0 - (tx_end_length - station_spacing)
    x1_tx = 0.
    y1_tx = length + tx_end_length
    stns_y_4_tx = np.arange(y0_tx, y1_tx, station_spacing)
    # update x stations
    stns_x_4_tx = np.zeros(stns_y_4_tx.shape[0])
    # now do tx lines
    num_lines = int(num_lines)
    spacing = float(line_spacing)

    stns_x_tx = np.zeros((stns_x_4_tx.shape[0], num_lines))
    stns_y_tx = np.array([stns_y_4_tx,]*num_lines).transpose()

    # now construct them
    dist = 0
    for ii in range(num_lines):
        if ii == 0:
            stns_x_tx[:, ii] = stns_x_4_tx + dist
            dist += spacing
        else:
            stns_x_tx[:, ii] = stns_x_4_tx + dist
            dist += spacing

    # now do rx lines
    num_lines = int(num_lines)
    spacing = float(line_spacing)

    # now for each line ========
    lines = []
    print(stns_x_tx.shape[1])
    for line in range(stns_x_tx.shape[1]):
        # get end points
        x0, y0 = 0., 0.
        x1, y1 = 0., length
        # get number of station
        number_of_stns = num_stations
        station_spacing = length / number_of_stns
        stns_y_ = stns_y_tx[:, line]
        stns_x_ = stns_x_tx[:, line]

        # now do rx stations
        num_lines_lead = int(num_active_lead)
        num_lines_behind = int(num_active_behind)
        spacing = float(line_spacing)

        # get end points
        # bounds
        lim_min = stns_x_tx.min()
        lim_max = stns_x_tx.max()
        stns_y = np.arange(y0, y1, station_spacing)
        if tx_rx_offset != 0:
            stns_y = stns_y + tx_rx_offset

        stns_x_rx_lead = np.zeros((stns_x.shape[0], num_lines_lead))
        # stns_y_rx = np.zeros((stns_y.shape[0], num_lines))
        stns_y_rx_lead = np.tile(stns_y, num_lines_lead)
        stns_x_rx_behind = np.zeros((stns_x.shape[0], num_lines_behind))
        # stns_y_rx = np.zeros((stns_y.shape[0], num_lines))
        stns_y_rx_behind = np.tile(stns_y, num_lines_behind)
        # do the leading
        dist = 0
        tx_on_rx = 'no'
        for ii in range(num_lines_lead):
            if ii == 0:
                # check if tx lines and rx lines are coincident
                if tx_on_rx == 'no':
                    stns_x_rx_lead[:, ii] = stns_x_tx[:stns_x_rx_lead.shape[0], line] + spacing / 2.
                    dist += (spacing / 2. + spacing)
                else:
                    stns_x_rx_lead[:, ii] = stns_x_tx[:stns_x_rx_lead.shape[0], line]
                    dist += spacing
            else:
                stns_x_rx_lead[:, ii] = stns_x_tx[:stns_x_rx_lead.shape[0], line] + dist
                dist += spacing
        # do the trailing lines
        dist = 0
        for jj in range(num_lines_behind):
            if jj == 0:
                # check if tx - rx on coincident lines
                if tx_on_rx == 'no':
                    stns_x_rx_behind[:, jj] = stns_x_tx[:stns_x_rx_lead.shape[0], line] - spacing / 2.
                    dist += (spacing / 2. + spacing)
                else:
                    stns_x_rx_behind[:, jj] = stns_x_tx[:stns_x_rx_lead.shape[0], line] - spacing
                    dist += (spacing * 2)
            else:
                stns_x_rx_behind[:, jj] = stns_x_tx[:stns_x_rx_lead.shape[0], line] - dist
                dist += spacing

        # reshape the matricies
        stns_x_rx = np.hstack((stns_x_rx_lead, stns_x_rx_behind))
        stns_y_rx = np.hstack((stns_y_rx_lead, stns_y_rx_behind))
        stns_y_rx = stns_y_rx.reshape(stns_x_rx.shape, order='F')

        # check the limits
        x_allowed = []
        y_allowed = []
        # print('before: ', stns_x_rx.shape, stns_y_rx.shape)
        for col in range(stns_x_rx.shape[1]):
            if tx_on_rx == 'yes':
                if lim_min <= stns_x_rx[0, col] <= lim_max:
                    x_allowed.append(stns_x_rx[:, col])
                    y_allowed.append(stns_y_rx[:, col])
            else:
                if lim_min < stns_x_rx[0, col] < lim_max:
                    x_allowed.append(stns_x_rx[:, col])
                    y_allowed.append(stns_y_rx[:, col])
        stns_x_rx = np.hstack(x_allowed) 
        stns_y_rx = np.hstack(y_allowed) 

        tx = transmit()

        # here we trying to sort the tx spacing
        tx_a = float(tx_site_spacing)
        # take first station and increment till last
        last_idx = stns_x_tx[:, line].shape[0] - 1
        user_tx_y = np.arange(stns_y_tx[0, line], stns_y_tx[last_idx, line] + tx_a, tx_a)
        user_tx_x = np.zeros(user_tx_y.shape) + stns_x_tx[0, line]
        # tx.transmit = np.vstack((stns_x_tx[:, line], stns_y_tx[:, line]))
    #     print(user_tx_x.shape, tx_a, stns_y_tx[0, line], stns_y_tx[last_idx, line])
        tx.transmit = np.vstack((user_tx_x, user_tx_y))
        tx.receivers = np.vstack((stns_x_rx, stns_y_rx))
        lines.append(tx)
        # print('assign: ', len(lines))
        transmits = lines

    electrodes = []
    for ii in range(len(transmits)):
        tx = transmits[ii].transmit
        rx = transmits[ii].receivers
        transmits[ii].transmit = tx
        transmits[ii].receivers = rx

    # ax = plt.subplot(1,1,1)
    # for line in transmits:
    #     ax.plot(line.transmit[0, :], line.transmit[1, :], 'or')
    #     ax.plot(line.receivers[0, :], line.receivers[1, :], '.b')       
    # ax.axis("equal")
    # plt.show()

    #########################################################################
    # create the simpeg survey object
    # -------------------------------
    data_type = "DC"
    dipole_min = 50
    dipole_max = 250
    for ii in range(len(transmits)):
        transmits[ii].createDipoles(dipole_min, dipole_max)

    # now write to file
    src_lists = []
    for ii in range(len(transmits)):
        for jj in range(transmits[ii].transmit.shape[1]):
            rx1 = []
            rx2 = []
            for kk in range(len(transmits[ii].dipoles)):
                rx1.append([transmits[ii].dipoles[kk][0][0], transmits[ii].dipoles[kk][0][1], 500.0])
                rx2.append([transmits[ii].dipoles[kk][1][0], transmits[ii].dipoles[kk][1][1], 500.0])
            rx1 = np.asarray(rx1)
            rx2 = np.asarray(rx2)   
            tx = np.asarray([transmits[ii].transmit[:, jj][0], transmits[ii].transmit[:, jj][1], 500.0])
    #         tx2 = np.asarray([-500, -500, 500])
            electrodes.append(np.asarray(rx1))
            electrodes.append(np.asarray(rx2))
            electrodes.append(np.asarray(tx))
    #         electrodes.append(np.asarray(tx2))
            Rx = dc.receivers.Dipole(rx1, rx2)    # create dipole list
            src_lists.append(dc.sources.Pole([Rx], tx))

    # book keeping for electrodes
    electrodes = np.vstack(electrodes)

    survey_dc = dc.Survey(src_lists)          # creates the survey
    # check if data is IP
    if data_type == "IP":
        survey_dc = ip.from_dc_to_ip_survey(survey_dc, dim="3D")

    electrodes = utils.uniqueRows(electrodes)
    electrodes = electrodes[0]

    ##########################################################################
    # 3D Tree Mesh
    # ------------

    # Cell sizes
    csx, csy, csz = 25.0, 25.0, 25.0
    # Number of core cells in each direction
    ncx, ncy, ncz = 54, 54, 27
    # Number of padding cells to add in each direction
    npad = 5
    # Vectors of cell lengths in each direction with padding
    # hx = [(csx, npad, -1.5), (csx, ncx), (csx, npad, 1.5)]
    # hy = [(csy, npad, -1.5), (csy, ncy), (csy, npad, 1.5)]
    # hz = [(csz, npad, -1.5), (csz, ncz)]
    # Create mesh and center it
    # global_mesh = discretize.TreeMesh([hx, hy, hz], x0=[-650, -650, -200])
    # padLen = 800
    # padding_distance = np.r_[np.c_[padLen, padLen], np.c_[padLen, padLen], np.c_[padLen, padLen]]
    # global_mesh = meshutils.mesh_builder_xyz(electrodes, [hx, hy, hz], mesh_type='TREE',
    #                              base_mesh=mesh, padding_distance=padding_distance,
    #                              depth_core=1000)
    # h = [25, 25, 25]
    # padDist = np.ones((3, 2)) * 500
    # global_mesh = meshutils.mesh_builder_xyz(
    #     electrodes, h,
    #     padding_distance=padDist,
    #     mesh_type='TREE',
    #     depth_core=200
    # )
    # global_mesh = meshutils.refine_tree_xyz(global_mesh, electrodes,
    #                              method='surface', octree_levels=[10, 5, 3, 2],
    #                              finalize=True)

    zblock = [(10, 5), (7.5, 13), (5, 10), (3.5, 165), (5, 5), (7.5, 5), (10, 5), (15, 3),
              (20, 3), (30, 2), (40, 2), (50, 1), (75, 1), (100, 1), (200, 1), (200, 1)]

    global_mesh = create_tensor_tile(survey_dc, zblock, 0, topo=None, min_cell_size=25.0, z_max=500)

    #########################################################################
    # spheres Model Creation
    # -----------------------

    # Spheres parameters
    x0, y0, z0, r0 = 400.0, 400.0, 350, 100.0
    # x1, y1, z1, r1 = 6.0, 0.0, -3.5, 3.0

    # ln conductivity
    ln_sigback = np.log(1e-5)
    ln_sigc = np.log(1e-3)

    # Define model
    # Background
    mtrue = ln_sigback * np.ones(global_mesh.nC)

    # Conductive sphere
    csph = (
        np.sqrt(
            (global_mesh.gridCC[:, 0] - x0) ** 2.0
            + (global_mesh.gridCC[:, 1] - y0) ** 2.0
            + (global_mesh.gridCC[:, 2] - z0) ** 2.0
        )
    ) < r0
    mtrue[csph] = ln_sigc * np.ones_like(mtrue[csph])

    active_cells = global_mesh.gridCC[:, 2] < 501
    mtrue[~active_cells] = np.log(1e-8)
    mstart = np.ones(active_cells.sum()) * ln_sigback
    survey_dc.drape_electrodes_on_topography(global_mesh, active_cells, option='top')

    # fig = plt.figure(figsize=(6, 6))
    # ax = fig.add_subplot(111)
    # global_mesh.plotSlice(mtrue, grid=True, normal="z", ind=5, ax=ax)
    # ax.plot(electrodes[:, 0], electrodes[:, 1], '.r')

    # fig = plt.figure(figsize=(6, 6))
    # ax1 = fig.add_subplot(111)
    # global_mesh.plotSlice(mtrue, grid=True, normal="y", ind=25, ax=ax1)
    # plt.show()

    ##########################################################################
    # Synthetic data simulation
    # -------------------------
    #

    # Setup Problem with exponential mapping and Active cells only in the core mesh
    expmap = maps.ExpMap(global_mesh)
    mapactive = maps.InjectActiveCells(mesh=global_mesh, indActive=active_cells, valInactive=np.log(1e-8))
    mapping = expmap * mapactive
    simulation_g = dc.Simulation3DNodal(
        global_mesh, survey=survey_dc, sigmaMap=mapping, solver=solver, model=mstart
    )
    simulation_g.mesh = TensorMesh([1])  # Light dummy
    # del local_mesh,
    # local_map.local_mesh = None
    expmap.mesh = None
    mapactive.mesh = None
    global_data = simulation_g.make_synthetic_data(mtrue[active_cells], relative_error=0.05, noise_floor=1., add_noise=True)


    # # plot predicted data
    # plt.plot(global_data.dobs, '.')
    # plt.title('data - DC')
    # plt.show()



    # ==============================================================================
    # finally split the simulations
    # -----------------------------

    print("[INFO] Creating tiled simulations over sources: ", len(survey_dc.source_list))
    survey_dc.dobs = global_data.dobs
    survey_dc.std = np.abs(survey_dc.dobs * global_data.relative_error) + global_data.noise_floor
    start_ = time()
    local_misfits = []
    z_max = 500.0

    idx_start = 0
    idx_end = 0
    # do every 5 sources
    cnt = 0
    src_collect = []
    for ii, source in enumerate(survey_dc.source_list):
        source._q = None # need this for things to work
        if cnt == 74 or ii == len(survey_dc.source_list)-1:
            src_collect.append(source)
            idx_end = idx_end + source.receiver_list[0].nD
            dobs = survey_dc.dobs[idx_start:idx_end]
    #         print(dobs.shape, len(src_collect))
            delayed_misfit = create_tile_dc(
                        src_collect,  survey_dc.dobs[idx_start:idx_end],
                        survey_dc.std[idx_start:idx_end], global_mesh, active_cells, ii, z_max)
            local_misfits += [delayed_misfit]

            idx_start = idx_end
            cnt = 0
            src_collect = []
        else:
    #         print(idx_start, idx_end)
            src_collect.append(source)
            idx_end = idx_end + source.receiver_list[0].nD
            cnt += 1
    global_misfit = objective_function.ComboObjectiveFunction(
                    local_misfits
    )


    #====================================================================================
    # new implementation using the Combo Objective function and the dmis = dmis1 + dmis2
    # -----------------------------------------------------------------------------------
    #

    # make intital model
    use_preconditioner = True
    coolingFactor = 2
    coolingRate = 1
    beta0_ratio = 1e1

    # Map for a regularization
    regmap = maps.IdentityMap(nP=int(active_cells.sum()))
    # reg = regularization.Tikhonov(mesh, indActive=global_actinds, mapping=regmap)
    reg = regularization.Sparse(global_mesh, indActive=active_cells, mapping=regmap)
    reg.norms = np.c_[0, 2, 2, 2]
    print('[INFO] Getting things started on inversion...')
    # set alpha length scales


    opt = optimization.ProjectedGNCG(
        maxIter=15, upper=np.inf, lower=-np.inf,
        maxIterCG=10, tolCG=1e-4
    )
    invProb = inverse_problem.BaseInvProblem(global_misfit, reg, opt)

    print("Pre-computing Jmatrix and predicted_0")
    invProb.dpred = invProb.get_dpred(mstart, compute_J=True)
    beta = directives.BetaSchedule(
        coolingFactor=coolingFactor, coolingRate=coolingRate
    )
    irls = directives.Update_IRLS(f_min_change=1e-4, minGNiter=1)
    betaest = directives.BetaEstimate_ByEig(beta0_ratio=beta0_ratio, method="ratio")
    # target = directives.TargetMisfit()
    # target.target = survey_dc.nD
    save_model = directives.SaveUBCModelEveryIteration(mesh=global_mesh, mapping=mapactive, file_name="DC_", replace=False)
    save_pred = directives.SavePredictedEveryIteration(data=global_data, data_type='ubc_dc', file_name="DC_",
                                                       replace=False)

    # Need to have basice saving function
    if use_preconditioner:
        update_Jacobi = directives.UpdatePreconditioner()
        updateSensW = directives.UpdateSensitivityWeights(threshold=1e-8)
        directiveList = [
            save_pred, save_model, updateSensW, irls, beta, betaest, update_Jacobi
        ]
    else:
        directiveList = [
            beta, betaest, irls
        ]
    inv = inversion.BaseInversion(
        invProb, directiveList=directiveList)
    opt.LSshorten = 0.5
    opt.remember('xc')

    # Run Inversion ================================================================
    tc = time()
    minv = inv.run(mstart)
    rho_est = mapactive * minv
    # np.save('model_out.npy', rho_est)
    print(f"Runtime {time()-tc} sec")
    global_mesh.writeUBC('OctreeMesh-test.msh', models={
        'ubc.con': np.exp(rho_est),
        # 'sensW.con': np.exp(mapactive * np.log(wr)),
        'true.con': np.exp(mapactive * mtrue[active_cells])
    })
    # global_mesh.writeUBC('OctreeMesh-test.msh', models={})
    # global_mesh.writeUBC('OctreeMesh-test.msh', models={'sensW.con': np.exp(mapactive * np.log(updateSensW.wr))})
    global_data.dobs = np.hstack(invProb.dpred)
    DCutils.writeUBC_DCobs("Predicted.pre", global_data, 3, "surface")


if __name__ == '__main__':
    survey_type = 'dipole-dipole'
    run(survey_type=survey_type, plotIt=True)
