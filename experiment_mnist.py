import tensorflow as tf
import os
from settings import DATA_DIR, FLAGS,ROOT
from network import ConvLSTMNetwork
from trainers import Trainer
from data_loader import MnistLoader


def main(not_parsed_args):
    if len(not_parsed_args) > 1:
        print("Unknown args:%s" % not_parsed_args)
        exit()
    dpt=os.path.join(DATA_DIR,"frame20_seqs1000.npz")
    testp=os.path.join(DATA_DIR,"frame20_seqs100.npz")
    DL=MnistLoader(dpt,testp)
    md=ConvLSTMNetwork(FLAGS)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    sess = tf.Session(config=config, graph=md.graph)
    print('Sess created.')
    trainer=Trainer(DL,md,sess,FLAGS)
    trainer.train()
    sess.close()

if __name__ == '__main__':
    tf.app.run()
