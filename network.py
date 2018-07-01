import tensorflow as tf
from layers import ConvLSTM,FinalLayer
import copy
from util import util

class ConvLSTMNetwork:
    """ConvLSTM network for frame prediction"""

    def __init__(self, flags, model_name=""):
        self.hidden_features = [int(x) for x in flags.num_hidden.split(',')]
        self.layers=len(self.hidden_features)

        self.cnn_size=flags.cnn_size
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.batch_norm=flags.batch_norm
        self.lr_input = tf.placeholder(tf.float32, shape=[], name="LearningRate")
        self.beta1 = flags.beta1
        self.beta2 = flags.beta2
        self.epsilon = flags.epsilon
        self.loss=0

        # x input sequence, y output sequence ground truth
        self.x = tf.placeholder(tf.float32, shape=[None, None, None,None, flags.patch_size**2*flags.channel], name='x')
        self.y = tf.placeholder(tf.float32, shape=[None, None, None,None, flags.patch_size**2*flags.channel], name="y")
        self.y_=None

        self.name = self.get_model_name(model_name)
        self.build_graph()
        self.build_optimizer()
        self.graph = tf.get_default_graph()

    def build_graph(self):
        n_in_feature=self.x.shape[-1]
        n_out_feature=self.y.shape[-1]
        nseq=self.x.shape[1]

        self.y_=[None]*nseq
        # generate one frame for each input frame, share convLSTM layer between frames, link 1st layer hidden state and cell between frames
        H1,C1=None,None # 1st layer
        for t in range(nseq):
            # encoding
            forecast_layers=[None]*self.layers
            in_feat=n_in_feature
            conl=ConvLSTM(self.cnn_size)
            h,c=conl.output('enco_conv1',in_feat,self.hidden_features[0],x=self.x[:,t],h=H1,c=C1)
            in_feat=self.hidden_features[0]
            forecast_layers[0]=copy.deepcopy(conl)
            H1=h
            C1=c
            for n in range(1,self.layers):
                conl=ConvLSTM(self.cnn_size)
                h,c=conl.output('enco_conv%d'%(n+1),in_feat,self.hidden_features[n],h=h,c=c)
                forecast_layers[n]=copy.deepcopy(conl)
                in_feat=self.hidden_features[n]

            # forecasting
            H=[None]*self.layers
            total_out_feat=0
            for k in range(self.layers):
                h,c=forecast_layers[k].output('fore_conv%d'%(k+1),in_feat,self.hidden_features[k],h=h,c=c)
                in_feat=self.hidden_features[k]
                H[k]=h
                total_out_feat+=self.hidden_features[k]
            with tf.variable_scope("Concat"):
                H_concat=tf.concat(H,3,name='H_concat')
            finl=FinalLayer(1)
            h,_=finl.output('fina_conv',H_concat,total_out_feat,n_out_feature)
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

