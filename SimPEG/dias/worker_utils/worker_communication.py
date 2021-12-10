import os
import numpy as np
import shutil
import numcodecs
import json
from threading import Thread, local 
import socket
import select
import struct
import time
import sys

numcodecs.blosc.use_threads = False

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

def workerRequest(outputs, simlite, host, index):
    """
        A basic method for handling worker communications
    """
    
    # construct the request message to be sent to the worker
    message_to = simlite["request"] + "!" + json.dumps(simlite)
    
    # construct final message that contains server instruction for size of data
    msg = struct.pack('>I', len(message_to)) + message_to.encode('utf-8')
    
    # create client socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2)

    # connect to remote host
    try :
        s.connect((host.split(":")[0], int(host.split(":")[1])))

        # send that data
        s.sendall(msg)

        # initiate the listening control variable
        listening = True
        while listening:
            
            #create the socket list
            socket_list = [s]

            # Get the list sockets which are readable
            ready_to_read,ready_to_write,in_error = select.select(socket_list , [], [])

            for sock in ready_to_read:             
                if sock == s:
                    # incoming message from remote server, s
                    data = recv_msg(sock)

                    if not data :
                        print('\nDisconnected from chat server')
                    
                    # check what came back from server
                    else :
                        print("rx data and checking: ", data.decode('utf-8')[:25])
                        
                        # check if initialization is confirmed
                        if "init" in data.decode('utf-8'):
                            server_response = json.loads(data.decode('utf-8'))

                            if server_response["init"] == True:
                                listening = False
                                
                                # assign confirmation
                                outputs[index] = server_response["init"]
                        
                        # check if predicted data is being sent back
                        elif "residual" in data.decode('utf-8'):
                            print("in residuals to store")
                            server_response = json.loads(data.decode('utf-8'))
                            print("converted to json")
                            print(len(outputs), index)
                            # assign the data
                            outputs[index] = np.asarray(server_response["residual"])
                            print("outputs assigned")
                            listening = False
                            print("\n\n Assigned residuals to outputs")

                        # check if jvec data is being sent back
                        elif "deriv2" in data.decode('utf-8'):                    
                            server_response = json.loads(data.decode('utf-8'))
                            
                            # assign the data
                            outputs[index] = np.asarray(server_response["deriv2"])
                            listening = False

                        # check if jvec data is being sent back
                        elif "deriv" in data.decode('utf-8'):                    
                            server_response = json.loads(data.decode('utf-8'))
                            
                            # assign the data
                            outputs[index] = np.asarray(server_response["deriv"])
                            listening = False
            
        # close the socket
        s.close()

    except:
        print("break!", sys.exc_info()[0])
        print("connection to worker failed")