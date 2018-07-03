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
    dpt=os.path.join(DATA_DIR,FLAGS.data_file)
    testp=os.path.join(DATA_DIR,FLAGS.test_file)
    DL=MnistLoader(testp,FLAGS,dpt)
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
