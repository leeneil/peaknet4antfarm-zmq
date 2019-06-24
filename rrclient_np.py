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
          "p_skip": 0.66,
          "n_policy": n_policy,
          "n_train_push": 1,
          "skip_trained": False,
          "build_train_list": True,
          "project_name": project_name}
trainer = Trainer(params)


########################


context = zmq.Context()
print "Connecting to server..."
socket = context.socket(zmq.REQ)
socket.connect ("tcp://psanagpu116:5556")

payload = (None, 0)
message = (None, None)


while True:
    socket.send_pyobj(payload)

    if message[0] is None:
        pass
    elif message[0] == "train":
        trainer.train()
        payload = (trainer.get_grads(), trainer.delta)
    elif message[0] == "validate":
        trainer.validate()
    else:
        break    

    #  Get the reply.
    message = socket.recv_pyobj()
    print("Received latest model and command " + message[0])
 
    new_model = message[1]
    trainer.setup_peaknet(new_model)
    








