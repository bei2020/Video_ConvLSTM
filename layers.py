import tensorflow as tf
import math

class ConvLSTM:
    """A LSTM with convolutions instead of multiplications.
  
    Reference:
      Xingjian, S. H. I., et al. "Convolutional LSTM network: A machine learning approach for precipitation nowcasting." Advances in Neural Information Processing Systems. 2015.
    """

    def __init__(self, cnn_size,shape, cnn_stride=1, weight_init='', weight_dev=1.0, batch_norm=False, dropout_rate=1, is_training=False):

        self.name = ""
        self._forget = None
        self._input = None
        self._output = None
        self._cell = None
        self.batch=shape[0]
        self.height=shape[1]
        self.width=shape[2]

        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        self.cnn_size = cnn_size
        self.cnn_stride = cnn_stride
        self.weight_init = weight_init
        self.weight_dev = weight_dev
        self.is_training = is_training
        self.receptive_fields = 0

    def _create_weight(self, shape, stddev=0.01, kname='stddev', name='weight'):
        """Return filter. Default truncated Gaussian distribution (0,0.01)"""
        kern = tf.truncated_normal(shape=shape, stddev=stddev)
        if kname == "he":
            n = shape[0] * shape[1] * shape[2]
            stddev = math.sqrt(2.0 / n)
            kern = tf.truncated_normal(shape=shape,mean=0.0, stddev=stddev)

        return tf.Variable(kern, name=name)


    def _create_bias(self, shape, value=0.0, name=None):
        bias = tf.constant(value, shape=shape)

        if name is None:
            return tf.Variable(bias)
        else:
            return tf.Variable(bias, name=name)

    def _conv2d(self, input_tensor, weight, stride, use_bias=False, use_batch_norm=False, pre_name=""):
        output = tf.nn.conv2d(input_tensor, weight, strides=stride, padding="SAME",
                              name=pre_name + "_conv")

        if use_bias:
            bias = self._create_bias([weight.shape[-1]], name="conv_B")
            output = tf.add(output, bias, name=pre_name + "_add")
        if use_batch_norm:
            output = tf.layers.batch_normalization(output, training=self.is_training, name=pre_name+'_BN')
        return output

    def output(self,name,in_features,out_features,x=None,h=None, c=None):
        """Return hidden state and cell.
            x: input at step t, h: hidden state at last step, c: cell at last step
        """
        assert (x is not None or h is not None)
        with tf.name_scope(name):
            if h is None:
                h = tf.zeros(tf.shape(x), dtype=tf.float32)
                c = tf.zeros([self.batch,self.height,self.width,out_features], dtype=tf.float32)
            if x is None:
                x= tf.zeros(tf.shape(h), dtype=tf.float32)
            Wxi=self._create_weight([self.cnn_size, self.cnn_size, in_features, out_features], stddev=self.weight_dev,
                                             kname=self.weight_init, name='conv_Wxi')
            Whi=self._create_weight([self.cnn_size, self.cnn_size, in_features, out_features], stddev=self.weight_dev,
                                    kname=self.weight_init, name='conv_Whi')
            Wxf=self._create_weight([self.cnn_size, self.cnn_size, in_features, out_features], stddev=self.weight_dev,
                                    kname=self.weight_init, name='conv_Wxf')
            Whf=self._create_weight([self.cnn_size, self.cnn_size, in_features, out_features], stddev=self.weight_dev,
                                    kname=self.weight_init, name='conv_Whf')
            Wxc=self._create_weight([self.cnn_size, self.cnn_size, in_features, out_features], stddev=self.weight_dev,
                                    kname=self.weight_init, name='conv_Wxc')
            Whc=self._create_weight([self.cnn_size, self.cnn_size, in_features, out_features], stddev=self.weight_dev,
                                    kname=self.weight_init, name='conv_Whc')
            Wxo=self._create_weight([self.cnn_size, self.cnn_size, in_features, out_features], stddev=self.weight_dev,
                                    kname=self.weight_init, name='conv_Wxo')
            Who=self._create_weight([self.cnn_size, self.cnn_size, in_features, out_features], stddev=self.weight_dev,
                                    kname=self.weight_init, name='conv_Who')
            Wci=self._create_weight([1,self.height , self.width, out_features], stddev=self.weight_dev,
                                    kname=self.weight_init, name='conv_Wci')
            Wcf=self._create_weight([1,self.height , self.width, out_features], stddev=self.weight_dev,
                                    kname=self.weight_init, name='conv_Wcf')
            Wco=self._create_weight([1,self.height , self.width, out_features], stddev=self.weight_dev,
                                    kname=self.weight_init, name='conv_Wco')
            stride=[1, self.cnn_stride, self.cnn_stride, 1]
            print('x'*10,x.shape,'\\n Wxi',Wxi.shape,'\\n h ',h.shape,Whi.shape,'\\n c ', Wci.shape,c.shape)
            kkk=Wci*c
            self._input=tf.sigmoid(self._conv2d(x,Wxi,stride=stride)+self._conv2d(h,Whi,use_bias=True,stride=stride)+Wci*c)
            self._forget=tf.sigmoid(self._conv2d(x,Wxf,stride=stride)+self._conv2d(h,Whf,use_bias=True,stride=stride)+Wcf*c)
            self._cell=self._forget*c+self._input*tf.tanh(self._conv2d(x,Wxc,stride=stride)+self._conv2d(h,Whc,stride=stride,use_bias=True))
            self._output=tf.sigmoid(self._conv2d(x,Wxo,stride=stride)+self._conv2d(h,Who,use_bias=True,stride=stride)+Wco*self._cell)
            h=self._output*tf.tanh(self._cell)

            return  h,self._cell


class FinalLayer:

    def __init__(self, cnn_size, cnn_stride=1, weight_init='', weight_dev=1.0, activator='', batch_norm=True, dropout_rate=1, is_training=False):

        self.name = ""
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        self.cnn_size = cnn_size
        self.cnn_stride = cnn_stride
        self.weight_init = weight_init
        self.weight_dev = weight_dev
        self.activator=activator
        self.is_training = is_training
        self.receptive_fields = 0

    def _create_weight(self, shape, stddev=0.01, kname='stddev', name='weight'):
        """Return filter. Default truncated Gaussian distribution (0,0.01)"""
        kern = tf.truncated_normal(shape=shape, stddev=stddev)
        if kname == "he":
            n = shape[0] * shape[1] * shape[2]
            stddev = math.sqrt(2.0 / n)
            kern = tf.truncated_normal(shape=shape,mean=0.0, stddev=stddev)

        return tf.Variable(kern, name=name)


    def _create_bias(self, shape, value=0.0, name=None):
        bias = tf.constant(value, shape=shape)

        if name is None:
            return tf.Variable(bias)
        else:
            return tf.Variable(bias, name=name)

    def _conv2d(self, input_tensor, weight, stride=1, use_bias=False, pre_name=""):
        output = tf.nn.conv2d(input_tensor, weight, strides=stride, padding="SAME",
                              name=pre_name + "_conv")

        if use_bias:
            bias = self._create_bias([weight.shape[-1]], name="conv_B")
            output = tf.add(output, bias, name=pre_name + "_add")
        if self.batch_norm:
            output = tf.layers.batch_normalization(output, training=self.is_training, name=pre_name+'_BN')
        return output

    def output(self, name, input,  in_features, out_features):
        """Return output tensor of one unit hidden layer and weight, bias"""
        with tf.name_scope(name):
            w=self._create_weight([self.cnn_size,self.cnn_size,in_features,out_features])
            out=self._conv2d(input,w)
            return out

