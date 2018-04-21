import tensorflow as tf
import numpy as np
import os

from utils import load_predict
from model import deepEM
from args import Predict_Args


def predict():
    args = Predict_Args()
    # test_index = [[mic_num, x, y]]
    test_x, test_index  = load_predict(args)
    checkpoint_dir = "./save"
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
#    gpu_config = tf.ConfigProto()
#    gpu_config.gpu_options.allow_growth = True
#    with tf.Session(config = gpu_config) as sess:
    checkpoint_dir = "./save"
    
    with tf.Session() as sess:
        deepem = deepEM(args)
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('Restore model failed!')
        output_pred = ""

        logit = tf.nn.softmax( deepem.logits) 

        pred = sess.run(logit, feed_dict = {deepem.X: test_x})

        for i in xrange(args.end_mic_num - args.start_mic_num):
            output_dir = './result/' + args.name_prefix + test_index[i][0] + '.box'
            output = open(output_dir, 'w')
            if pred[i] == 1:
                output.write(str(test_index[i][1])+'\t'+str(test_index[i][2])+'\t'+args.boxsize+'\t'+args.boxsize)

        output.close
                

        


if __name__ == '__main__':
    predict()

