import os
import datetime
from settings import DATA_DIR
import logging
import tensorflow as tf
from settings import FLAGS
import numpy as np
from scipy import misc

def preprocess_data():
    """split name to file dir"""
    fnames=os.listdir(DATA_DIR)
    fs = fnames[0].split('%2F')
    file_dir = os.path.join(DATA_DIR, fs[0])
    if not os.path.isdir(file_dir):
        os.mkdir(file_dir)
    if not os.path.isdir(os.path.join(file_dir, fs[1])):
        os.mkdir(os.path.join(file_dir, fs[1]))
    for f in fnames:
        fs = f.split('%2F')
        if len(fs) == 3:
            os.rename(os.path.join(DATA_DIR, f), os.path.join(DATA_DIR, fs[0], fs[1], fs[2]))


def array2image(dat,dest,post='',step=0):
    b,s,h,w,_=dat.shape
    dat=dat.reshape((b,s,h*FLAGS.patch_size,w*FLAGS.patch_size))
    _,_,h,w=dat.shape
    if not os.path.exists(os.path.join(dest,str(step))):
        os.makedirs(os.path.join(dest,str(step)))
    for i in range(b):
        if not os.path.exists(os.path.join(dest,str(step),str(i+1))):
            os.makedirs(os.path.join(dest,str(step),str(i+1)))
        for j in range(s):
            image = misc.toimage(dat[i,j].astype(np.uint8), cmin=0, cmax=255)  # to avoid range rescaling
            misc.imsave(os.path.join(dest,str(step),str(i+1), '{}_{}.jpg'.format(post,j+1)), image)

def print_num_of_total_parameters(output_detail=False, output_to_logging=False):
    total_parameters = 0
    parameters_string = ""

    for variable in tf.trainable_variables():

        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
        if len(shape) == 1:
            parameters_string += ("%s %d, " % (variable.name, variable_parameters))
        else:
            parameters_string += ("%s %s=%d, " % (variable.name, str(shape), variable_parameters))
    logging.info(
        "Total %d variables, %s params" % (len(tf.trainable_variables()), "{:,}".format(total_parameters)))

    if output_to_logging:
        if output_detail:
            logging.info(parameters_string)
    else:
        if output_detail:
            print(parameters_string)
        print("Total %d variables, %s params" % (len(tf.trainable_variables()), "{:,}".format(total_parameters)))

def get_now_date():
    d = datetime.datetime.today()
    return "%s/%s/%s %s:%s:%s" % (d.year, d.month, d.day, d.hour, d.minute, d.second)

def set_logging(filename, stream_log_level, file_log_level, tf_log_level):
    stream_log = logging.StreamHandler()
    stream_log.setLevel(stream_log_level)

    file_log = logging.FileHandler(filename=filename)
    file_log.setLevel(file_log_level)

    logger = logging.getLogger()
    logger.handlers = []
    logger.addHandler(stream_log)
    logger.addHandler(file_log)
    logger.setLevel(min(stream_log_level, file_log_level))

    tf.logging.set_verbosity(tf_log_level)
