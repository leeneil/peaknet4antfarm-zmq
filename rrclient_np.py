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
from trainer import Trainer


### Trainer setup ###

project_name = "runs/antfarm_zmq_b/1cls/ada/lr_0.001"
n_save = 10240 * 32
n_check = 128 * 32
n_policy = [32*200000, 32*400000, 32*800000]
init_lr = 0.001
macro_batch_size = 3
algo = "adagrad"
params = {"lr": init_lr, "macro_batch_size": macro_batch_size, 
          "optim": algo, "n_save": n_save, "n_check": n_check,
          "p_skip": 0.66, # depreciated
          "n_policy": n_policy, # at which iteration number the learning rate is reduced
          "n_train_push": 1, # how oftn the new grad is pushed
          "skip_trained": False, # if True, an exp-run will only be seen once
          "build_train_list": True, # if True, the trianing list is dynamically built
          "project_name": project_name}
trainer = Trainer(params)


########################


context = zmq.Context()
print "Connecting to server..."
socket = context.socket(zmq.REQ)
socket.connect ("tcp://psanagpu116:5556") # this should point to the server

payload = (None, 0)      # payload: client=>server ;  payload[0] = grads  , payload[1] = grads
message = (None, None)   # message: server=>client ;  message[0] = command, message[1] = model


while True:
    socket.send_pyobj(payload) # send payload to server even if there is nothing to send

    if message[0] is None: # if server is not quite ready, do nothing
        pass
    elif message[0] == "train":
        trainer.train()
        payload = (trainer.get_grads(), trainer.delta)
    elif message[0] == "validate":
        trainer.validate()   # validation mode doesn't send anything to the server
        payload = (None, 0)
    else:
        break    

    #  Get the reply.
    message = socket.recv_pyobj()
    print("Received latest model and command " + message[0]) # print out the latest command
 
    new_model = message[1]
    trainer.setup_peaknet(new_model) # update model
    








