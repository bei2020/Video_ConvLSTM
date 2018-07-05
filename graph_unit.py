import tensorflow as tf
import math


class ConvLSTM:
    """A LSTM with convolutions instead of multiplications.
  
    Reference:
      Xingjian, S. H. I., et al. "Convolutional LSTM network: A machine learning approach for precipitation nowcasting." Advances in Neural Information Processing Systems. 2015.
    """

    def __init__(self, cnn_size, shape, hidden_feature, cnn_stride=1, weight_init='', weight_dev=1.0, batch_norm=False,
                 dropout_rate=1, is_training=False, layer_name='',initializer=0.001):
        self.name = layer_name
        self._forget = None
        self._input = None
        self._output = None
        self._cell = tf.zeros([1, shape[1], shape[2], hidden_feature], dtype=tf.float32)
        self.height = shape[1]
        self.width = shape[2]
        self.out_features = hidden_feature
        # self.batch=shape[0]

        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        self.cnn_size = cnn_size
        self.cnn_stride = cnn_stride
        self.weight_init = weight_init
        self.weight_dev = weight_dev
        self.is_training = is_training
        self.receptive_fields = 0
        self.initializer = tf.random_uniform_initializer(-initializer,initializer)

    def _create_weight(self, shape, stddev=0.01, kname='stddev', name='weight'):
        """Return filter. Default truncated Gaussian distribution (0,0.01)"""
        kern = tf.truncated_normal(shape=shape, stddev=stddev)
        if kname == "he":
            n = shape[0] * shape[1] * shape[2]
            stddev = math.sqrt(2.0 / n)
            kern = tf.truncated_normal(shape=shape,mean=0.0, stddev=stddev)

        return tf.Variable(kern, name=name)

    def output(self, x, h):
        if h is None:
            h = tf.zeros((1, self.height, self.width, self.out_features), dtype=tf.float32)

        with tf.variable_scope(self.name):
            Wci=tf.get_variable('inputc',(1, self.height, self.width, self.out_features),dtype=tf.float32,initializer=self.initializer)
            Wcf=tf.get_variable('forgetc',(1, self.height, self.width, self.out_features),dtype=tf.float32,initializer=self.initializer)
            Wco=tf.get_variable('outputc',(1, self.height, self.width, self.out_features),dtype=tf.float32,initializer=self.initializer)

            self._input = tf.sigmoid(
                tf.layers.conv2d(x, self.out_features, self.cnn_size, padding='SAME',kernel_initializer=self.initializer,
                                 name='inputx2state',use_bias=False)
                + tf.layers.conv2d(h, self.out_features, self.cnn_size, padding='SAME',use_bias=True,kernel_initializer=self.initializer,
                                                                        name='inputh2state')+ Wci * self._cell)
            self._forget = tf.sigmoid(
                tf.layers.conv2d(x, self.out_features,self.cnn_size,padding='SAME',use_bias=False,kernel_initializer=self.initializer, name='forgetx2state')
                + tf.layers.conv2d(h, self.out_features,self.cnn_size,padding='SAME',use_bias=True,kernel_initializer=self.initializer, name='forgeth2state')
                + Wcf * self._cell)
            self._cell = self._forget * self._cell + self._input * tf.tanh(
                tf.layers.conv2d(x, self.out_features,self.cnn_size,padding='SAME',use_bias=False,kernel_initializer=self.initializer, name='cellx2state') +
                tf.layers.conv2d(h,self.out_features,self.cnn_size,padding='SAME',use_bias=True,kernel_initializer=self.initializer, name='cellh2state'))
            self._output = tf.sigmoid(
                tf.layers.conv2d(x, self.out_features,self.cnn_size,padding='SAME',use_bias=False,kernel_initializer=self.initializer, name='outputx2state')
                + tf.layers.conv2d(h,self.out_features,self.cnn_size,padding='SAME',use_bias=True,kernel_initializer=self.initializer, name='outputh2state')
                + Wco * self._cell)
            h = self._output * tf.tanh(self._cell)

            return h


class FinalLayer:

    def __init__(self, cnn_size, cnn_stride=1, weight_init='', weight_dev=1.0, activator='', batch_norm=True, dropout_rate=1, is_training=False,layer_name=''):

        self.name = layer_name
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        self.cnn_size = cnn_size
        self.cnn_stride = cnn_stride
        self.weight_init = weight_init
        self.weight_dev = weight_dev
        self.activator=activator
        self.is_training = is_training
        self.receptive_fields = 0

    def output(self, x,out_features):
        h=tf.layers.conv2d(x,out_features,self.cnn_size,self.cnn_stride,padding='SAME',name='finl_predict')
        return h