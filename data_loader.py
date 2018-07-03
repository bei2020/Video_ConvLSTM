import numpy as np
import random


class MnistLoader:
    def __init__(self,test_path,flags,train_path=None):
        self.x=None
        self.y=None
        self.x_test=None
        self.y_test=None
        self.train_path=train_path
        self.test_path=test_path
        self.batch_size_test=flags.batch_size_test
        self.seq_len=flags.seq_len
        self.patch_size=flags.patch_size
        self.channel=flags.channel

    def load_train(self):
        """Return input x and y, both are numpy.array([batch,seqs,height,width,channel])"""
        self.x,self.y=self.load_file(self.train_path)

    def load_test(self):
        """Return test x and y, both are numpy.array([batch,seqs,height,width,channel])"""
        self.x_test,self.y_test=self.load_file(self.test_path,self.batch_size_test)

    def load_file(self,file_path,batch_size=-1):
        dt=np.load(file_path)['arr_0']
        seq_len=self.seq_len
        nseq,_,height,width=dt.shape
        dt=dt.reshape((nseq//seq_len,seq_len,height//self.patch_size,width//self.patch_size,self.patch_size**2*self.channel))
        x=dt[:,:10][:,::-1]
        y=dt[:,10:]
        if batch_size>0:
            x=x[:batch_size]
            y=y[:batch_size]
        return x,y

def rand_seqs_idx(nscene):
    """Return randomized sequences locations"""
    return random.sample(range(nscene), nscene)

