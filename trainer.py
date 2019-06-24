import pandas as pd
import numpy as np
import glob
import os
import re
import h5py
import time
import psana
import torch
from peaknet import Peaknet
from peaknet_utils import *
from antfarm_utils import *


class Trainer(object):
    
    def __init__(self, params):
        self.params = params
        # get val list
        self.get_val_list()
        # get training list
        self.get_train_list()
        # set-up Peaknet
        self.setup_peaknet()
        self.grad = None
        self.delta = 0
        self.psana_ready = False
        self.cxi_ready = False
    
    def get_train_list(self):
        if self.params["build_train_list"]:
            self.df_train = get_train_df( cxi_path="/reg/d/psdm/cxi/cxitut13/res/autosfx", 
                                  val_csv="/reg/d/psdm/cxi/cxic0415/res/liponan/peaknet4antfarm/df_val.csv",
                                  test_csv="/reg/d/psdm/cxi/cxic0415/res/liponan/peaknet4antfarm/df_test.csv")
        else:
            self.df_train = pd.read_csv("/reg/d/psdm/cxi/cxic0415/res/liponan/peaknet4antfarm/df_train.csv", index_col=0)
        print("training list", len(self.df_train))
        
    def get_val_list(self, n=1000):
        self.df_val = pd.read_csv("/reg/d/psdm/cxi/cxic0415/res/liponan/peaknet4antfarm/df_val_events_1000.csv")
        self.df_val = self.df_val.sort_values(by=["exp", "run", "event"])
        print("validation list", len(self.df_val), "events")
        self.df_val_runs = self.df_val[["exp", "run", "path"]].drop_duplicates()
        print("validation list", len(self.df_val_runs), "runs")
        
    def setup_peaknet(self, model=None):
        self.net = Peaknet()
        self.net.loadCfg( "/reg/neh/home/liponan/ai/pytorch-yolo2/cfg/newpeaksv10-asic.cfg" )
        if model is None:
            self.net.init_model()
        else:
            self.net.model = model
        self.net.model.cuda()
        self.net.set_writer(project_name=self.params["project_name"], parameters=self.params)

    def get_grads(self):
        return self.net.getGrad()
    
    def validate(self):
        print("=========================================== VAL ===========================================")
        macro_batch_size = self.params["macro_batch_size"]
        seen = 0
        overall_recall = 0
        # validation
        for i in range(len(self.df_val_runs)):
            exp, run, path = self.df_val_runs.iloc[i][["exp", "run", "path"]]
            try:
                ds = psana.DataSource("exp=" + exp + ":run=" + str(run) + ":idx")
                det = psana.Detector('DscCsPad')
                this_run = ds.runs().next()
                times = this_run.times()
                print("*********************** {}-{} OKAY ***********************".format(exp, run))
                
            except:
                print("{}-{} not avaiable".format(exp, run))
                continue
            sub_events = self.df_val.query("exp == '{}' and run == '{}'".format(exp,run))["event"]
#             print(sub_events)
#             print("path", path)
            labels, eventIdxs = load_cxi_labels_yxhw( path, total_size=-1 )
            labels = [labels[i] for i in range(len(labels)) if eventIdxs[i] in sub_events]
            eventIdxs = [eventIdxs[i] for i in range(len(eventIdxs)) if eventIdxs[i] in sub_events]
            print("labels", len(labels), "eventIdxs", len(eventIdxs))
            n_iters = int( np.ceil( len(labels) / float(macro_batch_size) ) )
            print("# iterations", n_iters)

            for j in range(n_iters):
                idx_offset = j * macro_batch_size
                if j == (n_iters-1):
                    n = len(labels) - j*macro_batch_size
                    batch_imgs = psana_img_loader(eventIdxs, idx_offset, n, det, this_run, times)
                    batch_labels = labels[(j*macro_batch_size):]
                else:
                    n = macro_batch_size
                    batch_imgs = psana_img_loader(eventIdxs, idx_offset, macro_batch_size, det, this_run, times)
                    batch_labels = labels[j*macro_batch_size:(j+1)*macro_batch_size]
                batch_imgs[ batch_imgs < 0 ] = 0
                batch_imgs = batch_imgs / batch_imgs.max()
                my_recall = self.net.validate( batch_imgs, batch_labels, mini_batch_size=macro_batch_size*32 )
                print("my recall", my_recall)
                overall_recall += n * my_recall
                seen += n
        overall_recall /= (1.0*seen)
        self.net.writer.add_scalar('recall_val', overall_recall, self.net.model.seen)
        print("----------------------------------------- END VAL -----------------------------------------")

    def train(self):
        # params
        macro_batch_size = self.params["macro_batch_size"]
        algo = self.params["optim"]
        my_lr = self.params["lr"]
        n_check = self.params["n_check"]
        n_save = self.params["n_save"]
        n_policy = self.params["n_policy"]
        skip_trained = self.params["skip_trained"]
        p_skip = self.params["p_skip"]
        n_train_push = self.params["n_train_push"]
        # training
        #self.nets[0].set_writer(project_name=self.params["project_name"], parameters=self.params)
        #self.nets[0].writer.add_scalar('lr', my_lr, self.nets[0].model.seen)
        
        
        while not self.psana_ready:
            self.exp, self.run, self.path = self.df_train.sample(1).iloc[0][["exp", "run", "path"]]
            #self.exp = "cxic0415"
            #self.run = '91'
            #self.path = "/reg/d/psdm/cxi/cxitut13/res/autosfx/cxic0415_0091.cxi"
            #print(exp, run, path)
            time.sleep(1)
            try:
                self.ds = psana.DataSource("exp=" + self.exp + ":run=" + str(self.run) + ":idx")
                #print(self.ds)
                self.det = psana.Detector('DscCsPad')  #FIXME: could be other CsPad?
                #print(self.det)
                self.this_run = self.ds.runs().next()
                #print(self.this_run)
                self.times = self.this_run.times()
                print("*********************** {}-{} OKAY ***********************".format(self.exp, self.run))                
            except:
                print("{}-{} not avaiable".format(self.exp, self.run))
                continue  
            if skip_trained:
                log_filename = os.path.join("/reg/d/psdm/cxi/cxic0415/res/liponan/peaknet4antfarm/train_log", "{}_{}".format(exp,run))
                if os.path.isfile( log_filename ):
                    continue
                else:
                    with open(log_filename, 'a'):
                        os.utime(log_filename, None)
            self.psana_ready = True 
            self.j_iter = 0
            print("end of psana test")
                
        #self.net.writer.add_text("EXP-RUN", "{}-{}".format(exp, run), self.net.model.seen)

        if not self.cxi_ready:
            self.labels, self.eventIdxs = load_cxi_labels_yxhw( self.path, total_size=-1 )
            print("labels", len(self.labels), "eventIdxs", len(self.eventIdxs))
            self.n_iters = int( np.floor( len(self.labels) / float(macro_batch_size) ) )
            print("# iterations", self.n_iters)

        self.net.set_optimizer(adagrad=(algo=="adagrad"), lr=my_lr )
            
        for j in range(self.j_iter, self.j_iter+n_train_push): # was n_iters
            self.delta = n_train_push
            if self.j_iter == self.n_iters-1:
                self.psana_ready = False
                self.cxi_ready = False    
                
            idx_offset = j * macro_batch_size
            n = macro_batch_size
            batch_imgs = psana_img_loader(self.eventIdxs, idx_offset, macro_batch_size, self.det, self.this_run, self.times)
            batch_labels = self.labels[j*macro_batch_size:(j+1)*macro_batch_size]
            self.net.set_optimizer(adagrad=(algo=="adagrad"), lr=my_lr )
            batch_imgs[ batch_imgs < 0 ] = 0
            batch_imgs = batch_imgs / batch_imgs.max()
            self.net.train( batch_imgs, batch_labels, mini_batch_size=macro_batch_size*32 )
            self.grad = self.net.getGrad()
                
            if self.net.model.seen % n_save == 0:
                self.net.snapshot(batch_imgs, batch_labels, tag="antfarm_zmq_trainer")
                print("snapshot saved")
                    
            if self.net.model.seen in n_policy:
                my_lr /= 10.0
                self.net.writer.add_scalar('lr', my_lr, self.net.model.seen)
            self.j_iter += 1
                    
    
    


