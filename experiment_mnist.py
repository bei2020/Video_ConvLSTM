import tensorflow as tf
import os
from data_loader import loader
from settings import DATA_DIR, FLAGS,ROOT
from network import ConvLSTMNetwork
from util.util import array2image
import logging

class Trainer:
    def __init__(self,data,model,sess,flags):
        self.data=data
        self.model=model
        self.sess=sess

        self.lr = flags.lr
        self.lr_decay = flags.lr_decay
        self.lr_decay_epoch = flags.lr_decay_epoch
        self.end_lr = flags.end_lr
        self.batch_size=flags.batch_size
        self.epoch_evaluate=flags.epoch_evaluate

        self.output_dir=flags.output_dir

        self.build_summary_saver()
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)


    def train_batch(self,batch_x,batch_y):
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y}
        _, loss = self.sess.run([self.model.training_optimizer, self.model.loss], feed_dict=feed_dict)

    def train(self,test_file):
        x_ims,y_ims=self.data
        nbatches=x_ims.shape[0]
        total_batch=nbatches//self.batch_size+(0,1)[nbatches%self.batch_size>0]
        nbatch=0

        while not(self.lr<self.end_lr or nbatch>=total_batch):
            self.train_batch(x_ims[nbatch:nbatch+8],y_ims[nbatch:nbatch+8])
            nbatch+=1
            if nbatch%self.lr_decay_epoch==0:
                self.lr*=self.lr_decay
            if nbatch%self.epoch_evaluate==0:
                self.evaluate(test_file)

        self.test(test_file)
        self.save_model()

    def evaluate(self,test_file):
        x_ims,y_ims=loader(test_file,batch_size=1)
        feed_dict = {self.model.x: x_ims, self.model.y: y_ims}
        y_=self.sess.run(self.model.y_,feed_dict=feed_dict)
        loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_,labels=y_ims))
        print('Loss:%f'%loss)
        return

    def test(self,test_file):
        x_ims,y_ims=loader(test_file,batch_size=1)
        feed_dict = {self.model.x: x_ims, self.model.y: y_ims}
        y_=self.sess.run(self.model.y_,feed_dict=feed_dict)
        loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_,labels=y_ims))
        print('Loss:%f'%loss)
        array2image(y_,os.path.join(ROOT,self.output_dir),'predict')
        array2image(y_ims,os.path.join(ROOT,self.output_dir),'truth')
        return

    def save_model(self, name="", trial=0, output_log=False):
        if name == "" or name == "default":
            name = self.model.name
        if trial > 0:
            filename = self.checkpoint_dir + "/" + name + "_" + str(trial) + ".ckpt"
        else:
            filename = self.checkpoint_dir + "/" + name + ".ckpt"
        self.saver.save(self.sess, filename)
        if output_log:
            logging.info("Model saved [%s]." % filename)
        else:
            print("Model saved [%s]." % filename)

    def build_summary_saver(self):
        if self.save_loss or self.save_weights or self.save_meta_data:
            self.summary_op = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(self.tf_log_dir + "/train")
            self.test_writer = tf.summary.FileWriter(self.tf_log_dir + "/test", graph=self.sess.graph)

        self.saver = tf.train.Saver(max_to_keep=None)


def main(not_parsed_args):
    if len(not_parsed_args) > 1:
        print("Unknown args:%s" % not_parsed_args)
        exit()
    dpt=os.path.join(DATA_DIR,"frame20_seqs1000.npz")
    testp=os.path.join(DATA_DIR,"frame20_seqs100.npz")
    x_ims,y_ims=loader(dpt)
    md=ConvLSTMNetwork(FLAGS)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    sess = tf.Session(config=config, graph=md.graph)
    print('Sess created.')
    trainer=Trainer((x_ims,y_ims),md,sess,FLAGS)
    trainer.train(testp)

if __name__ == '__main__':
    tf.app.run()
