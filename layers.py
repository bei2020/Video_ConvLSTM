import tensorflow as tf
import math

class ConvLSTM:
    """A LSTM with convolutions instead of multiplications.
  
    Reference:
      Xingjian, S. H. I., et al. "Convolutional LSTM network: A machine learning approach for precipitation nowcasting." Advances in Neural Information Processing Systems. 2015.
      
    ! Caution: tf.Variable must be replaced with tf.get_variable for reuse under a name scope.
    """

    def __init__(self, cnn_size,shape,hidden_feature, cnn_stride=1, weight_init='', weight_dev=1.0, batch_norm=False, dropout_rate=1, is_training=False,layer_name=''):

        self.name = layer_name
        self._forget = None
        self._input = None
        self._output = None
        self._cell = tf.zeros([1,shape[1],shape[2],hidden_feature], dtype=tf.float32)
        self.height=shape[1]
        self.width=shape[2]
        self.out_features=hidden_feature
        # self.batch=shape[0]

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


    def _create_bias(self, shape, value=0.0, name='bias'):
        bias = tf.constant(value, shape=shape)

        if name is None:
            return tf.Variable(bias)
        else:
            return tf.Variable(bias, name=name)

    def _conv2d(self, input_tensor, weight, stride, use_bias=False, use_batch_norm=False, pre_name=""):
        output = tf.nn.conv2d(input_tensor, weight, strides=stride, padding="SAME", name=pre_name)

        if use_bias:
            bias = self._create_bias([weight.shape[-1]],name=pre_name+'_b')
            output = tf.add(output, bias, name=pre_name + "_add")
        if use_batch_norm:
            output = tf.layers.batch_normalization(output, training=self.is_training, name=pre_name+'_BN')
        return output

    def output(self,x=None,h=None,in_features=0,in_features_h=0,reuse=False):
        """Return hidden state and cell.
            x: input at step t, h: hidden state of last layer, c: cell of last sequence
        """
        assert (x is not None or h is not None)
        print('re'*10,reuse , self.name)
        stride=[1, self.cnn_stride, self.cnn_stride, 1]
        with tf.variable_scope(self.name):
            Wxi=self._create_weight([self.cnn_size, self.cnn_size, in_features, self.out_features], stddev=self.weight_dev,
                                             kname=self.weight_init, name='conv_Wxi')
            Whi=self._create_weight([self.cnn_size, self.cnn_size, in_features_h, self.out_features], stddev=self.weight_dev,
                                    kname=self.weight_init, name='conv_Whi')
            Wxf=self._create_weight([self.cnn_size, self.cnn_size, in_features, self.out_features], stddev=self.weight_dev,
                                    kname=self.weight_init, name='conv_Wxf')
            Whf=self._create_weight([self.cnn_size, self.cnn_size, in_features_h, self.out_features], stddev=self.weight_dev,
                                    kname=self.weight_init, name='conv_Whf')
            Wxc=self._create_weight([self.cnn_size, self.cnn_size, in_features, self.out_features], stddev=self.weight_dev,
                                    kname=self.weight_init, name='conv_Wxc')
            Whc=self._create_weight([self.cnn_size, self.cnn_size, in_features_h, self.out_features], stddev=self.weight_dev,
                                    kname=self.weight_init, name='conv_Whc')
            Wxo=self._create_weight([self.cnn_size, self.cnn_size, in_features, self.out_features], stddev=self.weight_dev,
                                    kname=self.weight_init, name='conv_Wxo')
            Who=self._create_weight([self.cnn_size, self.cnn_size, in_features_h, self.out_features], stddev=self.weight_dev,
                                    kname=self.weight_init, name='conv_Who')
            Wci=self._create_weight([1,self.height , self.width, self.out_features], stddev=self.weight_dev,
                                    kname=self.weight_init, name='ele_Wci')
            Wcf=self._create_weight([1,self.height , self.width, self.out_features], stddev=self.weight_dev,
                                    kname=self.weight_init, name='ele_Wcf')
            Wco=self._create_weight([1,self.height , self.width, self.out_features], stddev=self.weight_dev,
                                    kname=self.weight_init, name='ele_Wco')
            if not reuse and h is None:
                print('xr'*20,x.get_shape().as_list(),in_features,self.out_features,in_features_h)
                h=tf.zeros((1,self.height,self.width,self.out_features))
                self._input=tf.sigmoid(self._conv2d(x,Wxi,stride=stride,pre_name='Wxi')+self._conv2d(h,Whi,use_bias=True,stride=stride,pre_name='Whi')+Wci*self._cell)
                self._forget=tf.sigmoid(self._conv2d(x,Wxf,stride=stride,pre_name='Wxf')+self._conv2d(h,Whf,use_bias=True,stride=stride,pre_name='Whf')+Wcf*self._cell)
                self._cell=self._forget*self._cell+self._input*tf.tanh(self._conv2d(x,Wxc,stride=stride,pre_name='Wxc')+self._conv2d(h,Whc,stride=stride,use_bias=True,pre_name='Whc'))
                self._output=tf.sigmoid(self._conv2d(x,Wxo,stride=stride,pre_name='Wxo')+self._conv2d(h,Who,use_bias=True,stride=stride,pre_name='Who')+Wco*self._cell)
            else:
                # print('x'*10,x.shape,'\\n Wxi',Wxi.shape,'\\n h ',h.shape,Whi.shape,'\\n c ',Wci.shape)
                if h is None:
                    # print('x'*20,x.get_shape().as_list(),in_features,out_features)
                    self._input=tf.sigmoid(self._conv2d(x,Wxi,stride=stride,pre_name='Wxi',use_bias=True)+Wci*self._cell)
                    self._forget=tf.sigmoid(self._conv2d(x,Wxf,stride=stride,pre_name='Wxf',use_bias=True)+Wcf*self._cell)
                    self._cell=self._forget*self._cell+self._input*tf.tanh(self._conv2d(x,Wxc,stride=stride,pre_name='Wxc',use_bias=True))
                    self._output=tf.sigmoid(self._conv2d(x,Wxo,stride=stride,pre_name='Wxo',use_bias=True)+Wco*self._cell)
                elif x is None:
                    # print('h'*20,h.get_shape().as_list(),in_features_h,out_features)
                    self._input=tf.sigmoid(self._conv2d(h,Whi,use_bias=True,stride=stride,pre_name='Whi')+Wci*self._cell)
                    self._forget=tf.sigmoid(self._conv2d(h,Whf,use_bias=True,stride=stride,pre_name='Whf')+Wcf*self._cell)
                    self._cell=self._forget*self._cell+self._input*tf.tanh(self._conv2d(h,Whc,stride=stride,use_bias=True,pre_name='Whc'))
                    self._output=tf.sigmoid(self._conv2d(h,Who,use_bias=True,stride=stride,pre_name='Who')+Wco*self._cell)
                else:
                    # print('xh'*20,x.get_shape().as_list(),in_features,out_features,in_features_h)
                    self._input=tf.sigmoid(self._conv2d(x,Wxi,stride=stride,pre_name='Wxi')+self._conv2d(h,Whi,use_bias=True,stride=stride,pre_name='Whi')+Wci*self._cell)
                    self._forget=tf.sigmoid(self._conv2d(x,Wxf,stride=stride,pre_name='Wxf')+self._conv2d(h,Whf,use_bias=True,stride=stride,pre_name='Whf')+Wcf*self._cell)
                    self._cell=self._forget*self._cell+self._input*tf.tanh(self._conv2d(x,Wxc,stride=stride,pre_name='Wxc')+self._conv2d(h,Whc,stride=stride,use_bias=True,pre_name='Whc'))
                    self._output=tf.sigmoid(self._conv2d(x,Wxo,stride=stride,pre_name='Wxo')+self._conv2d(h,Who,use_bias=True,stride=stride,pre_name='Who')+Wco*self._cell)
            h=self._output*tf.tanh(self._cell)

            return  h


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

    def _create_weight(self, shape, stddev=0.01, kname='stddev', name='weight'):
        """Return filter. Default truncated Gaussian distribution (0,0.01)"""
        kern = tf.truncated_normal(shape=shape, stddev=stddev)
        if kname == "he":
            n = shape[0] * shape[1] * shape[2]
            stddev = math.sqrt(2.0 / n)
            kern = tf.truncated_normal(shape=shape,mean=0.0, stddev=stddev)

        return tf.Variable(kern, name=name)


    def _create_bias(self, shape, value=0.0, name='bias'):
        bias = tf.constant(value, shape=shape)

        if name is None:
            return tf.Variable(bias)
        else:
            return tf.Variable(bias, name=name)

    def _conv2d(self, input_tensor, weight, stride, use_bias=False, pre_name=""):
        output = tf.nn.conv2d(input_tensor, weight, strides=stride, padding="SAME", name=pre_name+'_conv')

        if use_bias:
            bias = self._create_bias([weight.shape[-1]], name=pre_name + "_b")
            output = tf.add(output, bias, name=pre_name + "_add")
        if self.batch_norm:
            output = tf.layers.batch_normalization(output, training=self.is_training, name=pre_name+'_BN')
        return output

    def output(self, input,  in_features, out_features,reuse=False):
        """Return output tensor of one unit hidden layer and weight, bias"""
        # with tf.variable_scope(self.name):
            # print('f'*20,input.get_shape().as_list(),in_features,out_features)
        w=self._create_weight([self.cnn_size,self.cnn_size,in_features,out_features],name='Wfn')
        out=self._conv2d(input,w,[1, self.cnn_stride, self.cnn_stride, 1],pre_name='convfn')
        return out

