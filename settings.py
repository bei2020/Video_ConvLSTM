import os
import tensorflow as tf

DATA_DIR=os.path.join(os.getcwd(),'data')
ROOT=os.getcwd()

flags = tf.app.flags
FLAGS = flags.FLAGS

#model
flags.DEFINE_integer("cnn_size", 5, "Size of CNN filters")
# flags.DEFINE_string("weight_init", "he", "Initializer for weights can be [uniform, stddev, xavier, he, identity, zero]")
# flags.DEFINE_float("weight_dev", 0.01, "Initial weight stddev (won't be used when you use he or xavier initializer)")
flags.DEFINE_string('num_hidden', '128,64,64', 'COMMA separated number of features in a convlstm layer.')

# data
flags.DEFINE_integer('batch_size', 8, 'batch size for training.')
flags.DEFINE_integer('batch_size_test', 1, 'batch size for test.')
flags.DEFINE_integer('seq_len', 20, 'total input and output length.')
flags.DEFINE_integer('patch_size', 4, 'patch size on one dimension.')
flags.DEFINE_integer('channel', 1, 'image channel')
flags.DEFINE_integer('height', 64, 'image height')
flags.DEFINE_integer('width', 64, 'image width')
flags.DEFINE_string('data_file',"frame20_seqs1000.npz", 'train')
flags.DEFINE_string('test_file',"frame20_seqs100.npz", 'test')

#training
flags.DEFINE_float('lr', 0.001, 'initial learning rate.')
flags.DEFINE_float("lr_decay", 0.9, "Learning rate decay rate")
# flags.DEFINE_float("end_lr", 2e-5,
#                    "Training end learning rate. If the current learning rate gets lower than this value, then training will be finished.")
flags.DEFINE_float("end_lr", 2e-5,
                   "Training end learning rate. If the current learning rate gets lower than this value, then training will be finished.")
flags.DEFINE_integer("lr_decay_epoch", 9, "After this epochs are completed, learning rate will be decayed by lr_decay.")
flags.DEFINE_integer("epoch_evaluate", 30, "After this epochs are completed, evaluate loss.")
flags.DEFINE_boolean("batch_norm", True, "use batch normalization after each CNN layer")
flags.DEFINE_float("beta1", 0.9, "Beta1 for adam optimizer")
flags.DEFINE_float("beta2", 0.999, "Beta2 for adam optimizer")
flags.DEFINE_float("epsilon", 1e-8, "epsilon for adam optimizer")
flags.DEFINE_boolean("save_loss", True, "Save loss")
flags.DEFINE_boolean("save_weights", True, "Save weights and biases")
flags.DEFINE_boolean("save_meta_data", False, "")
flags.DEFINE_string('output_dir','img', 'generated image dir')
flags.DEFINE_string("checkpoint_dir", "model", "Directory for checkpoints")
flags.DEFINE_string("tf_log_dir", "tf_log", "Directory for tensorboard log")


flags.DEFINE_string("log_filename", "log.txt", "log filename")
