import tensorflow as tf
from layers import ConvLSTM,FinalLayer
import copy
from util import util

class ConvLSTMNetwork:
    """ConvLSTM network for frame prediction"""

    def __init__(self, flags, model_name=""):
        self.hidden_features = [int(x) for x in flags.num_hidden.split(',')]
        self.layers=len(self.hidden_features)

        # model
        self.cnn_size=flags.cnn_size
        self.channel=flags.patch_size**2*flags.channel
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.batch_norm=flags.batch_norm
        self.lr_input = tf.placeholder(tf.float32, shape=[], name="LearningRate")
        self.beta1 = flags.beta1
        self.beta2 = flags.beta2
        self.epsilon = flags.epsilon
        self.loss=0
        self.encoding=[None]*self.layers
        self.forecast=[None]*self.layers

        # x input sequence, y output sequence ground truth
        self.x = tf.placeholder(tf.float32, shape=[flags.batch_size,flags.seq_len//2 ,flags.height//flags.patch_size,
                                                   flags.width//flags.patch_size, flags.patch_size**2*flags.channel], name='x')
        self.y = tf.placeholder(tf.float32, shape=[flags.batch_size,flags.seq_len//2 ,flags.height//flags.patch_size,
                                                   flags.width//flags.patch_size, flags.patch_size**2*flags.channel], name="y")
        self.y_=None
        self.nseq=flags.seq_len//2

        self.name = self.get_model_name(model_name)
        self.build_graph()
        self.build_optimizer()
        self.graph = tf.get_default_graph()


    def build_graph(self):
        # initial layers
        shape=self.x.get_shape().as_list()
        # batch,height,width,channel
        shape=(shape[0],shape[2],shape[3],shape[4])
        for k in range(self.layers):
            self.encoding[k]=ConvLSTM(self.cnn_size,shape,batch_norm=self.batch_norm,is_training=self.is_training)
            self.encoding[k]._cell = tf.zeros([shape[0],shape[1],shape[2],self.hidden_features[k]], dtype=tf.float32)
            self.forecast[k]=ConvLSTM(self.cnn_size,shape,batch_norm=self.batch_norm,is_training=self.is_training)
            self.forecast[k]._cell = tf.zeros([shape[0],shape[1],shape[2],self.hidden_features[k]], dtype=tf.float32)
        finl=FinalLayer(1,batch_norm=self.batch_norm,is_training=self.is_training)

        n_in_feature=shape[-1]
        self.y_=[None]*self.nseq
        # generate one frame for each input frame, share convLSTM layer between frames, link 1st layer hidden state between frames
        H1=None # 1st layer
        total_out_feat=sum(self.hidden_features)
        for t in range(self.nseq):
            print('t'*10,t)
            # encoding
            h=self.encoding[0].output('enco_conl1',self.hidden_features[0],x=self.x[:,t],h=H1,in_features=n_in_feature,in_features_h=self.hidden_features[0])
            in_feat=self.hidden_features[0]
            H1=h
            for n in range(1,self.layers):
                print('i'*15,n,in_feat,self.hidden_features[n])
                h=self.encoding[n].output('enco_conl%d'%(n+1),self.hidden_features[n],h=h,in_features_h=in_feat)
                in_feat=self.hidden_features[n]

            # forecasting
            H=[None]*self.layers
            for k in range(self.layers):
                h=self.forecast[k].output('fore_conl%d'%(k+1),self.hidden_features[k],h=h,in_features_h=in_feat)
                in_feat=self.hidden_features[k]
                H[k]=h
            with tf.variable_scope("Concat"):
                H_concat=tf.concat(H,3,name='H_concat')

            h=finl.output('fina_conv',H_concat,total_out_feat,n_in_feature)
            self.y_[t]=h
        self.y_ = tf.transpose(tf.stack(self.y_), [1,0,2,3,4])

    def build_optimizer(self):
        self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.y_,labels=self.y))
        if self.batch_norm:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.training_optimizer = tf.train.AdamOptimizer(self.lr_input, beta1=self.beta1, beta2=self.beta2, epsilon=self.epsilon).minimize(self.loss)
        else:
            self.training_optimizer = tf.train.AdamOptimizer(self.lr_input, beta1=self.beta1, beta2=self.beta2, epsilon=self.epsilon).minimize(self.loss)

        util.print_num_of_total_parameters(output_detail=True)


    def get_model_name(self, model_name, name_postfix=""):
        if model_name is "":
            name = "NS_L%d" % (self.layers)
            if self.cnn_size != 5:
                name += "_C%d" % self.cnn_size
            if self.batch_norm:
                name += "_BN"
            if name_postfix is not "":
                name += "_" + name_postfix
        else:
            name = "NS_%s" % model_name

        return name

