import os
import numpy as np
from settings import FLAGS
import random


def loader(file_path,batch_size=-1):
    dt=np.load(file_path)['arr_0']
    seq_len=FLAGS.seq_len
    nseq,_,height,width=dt.shape
    dt=dt.reshape((nseq//seq_len,seq_len,height//FLAGS.patch_size,width//FLAGS.patch_size,FLAGS.patch_size**2*FLAGS.channel))
    x=dt[:,:10]
    y=dt[:,10:]
    x=x[:,::-1]
    if batch_size>0:
        x=x[:batch_size]
        y=y[:batch_size]
    return x,y

def rand_seqs_idx(nscene):
    """Return randomized sequences locations"""
    return random.sample(range(nscene), nscene)

