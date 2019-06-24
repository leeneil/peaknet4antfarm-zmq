import zmq
import sys
import os
sys.path.append(os.path.abspath("/reg/neh/home/liponan/ai/peaknet4antfarm"))
import pandas as pd
import numpy as np
import glob
import re
import h5py
import time
import psana
import torch
from peaknet import Peaknet
from peaknet_utils import *
from antfarm_utils import *

n_validate = 120 

### Peaknet setup ###

net = Peaknet()
net.loadCfg( "/reg/neh/home/liponan/ai/pytorch-yolo2/cfg/newpeaksv10-asic.cfg" )
net.init_model()
net.model
print("done model setup")

#####################

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5556")

while True:
    #  Wait for next request from client
    message = socket.recv_pyobj()
    grads, delta = message
    print("Received request. delta:", delta)
    #time.sleep (np.random.uniform(0,0.01))
    if delta > 0:
        net.set_optimizer(adagrad=True)
        net.updateGrad(grads=grads, delta=delta, useGPU=False)
        net.optimize()
    print("imgs seen:", net.model.seen)
    if net.model.seen % n_validate == 0 and net.model.seen > 0:
        socket.send_pyobj(["validate", net.model])
    else:
        socket.send_pyobj(["train", net.model])
