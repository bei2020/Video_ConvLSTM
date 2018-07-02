import os
import numpy as np
from settings import FLAGS
import random


class MnistLoader:
    def __init__(self,test_path,train_path=None):
        self.x=None
        self.y=None
        self.x_test=None
        self.y_test=None
        self.train_path=train_path
        self.test_path=test_path

    def load_train(self,size=-1):
        """Return input x and y, both are numpy.array([batch,seqs,height,width,channel])"""
        self.x,self.y=self.load_file(self.train_path,size)

    def load_test(self,size=-1):
        """Return test x and y, both are numpy.array([batch,seqs,height,width,channel])"""
        self.x_test,self.y_test=self.load_file(self.test_path,size)

    def load_file(self,file_path,batch_size=-1):
        dt=np.load(file_path)['arr_0']
        seq_len=FLAGS.seq_len
        nseq,_,height,width=dt.shape
        dt=dt.reshape((nseq//seq_len,seq_len,height//FLAGS.patch_size,width//FLAGS.patch_size,FLAGS.patch_size**2*FLAGS.channel))
        x=dt[:,:10][:,::-1]
        y=dt[:,10:]
        if batch_size>0:
            x=x[:batch_size]
            y=y[:batch_size]
        return x,y

def rand_seqs_idx(nscene):
    """Return randomized sequences locations"""
    return random.sample(range(nscene), nscene)

