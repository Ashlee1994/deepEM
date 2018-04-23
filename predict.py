import tensorflow as tf
import numpy as np
import mrcfile
import time
import os

from utils import load_predict
from utils import sub_img
from utils import mapstd
from utils import rotate_map
from model import deepEM
from KLH import Predict_Args


def predict():
    args = Predict_Args()
    deepem = deepEM(args)
    # test_index = [[mic_num, x, y]]
    print("read test data start.")
    #read_data = time.time()
    #test_x, test_index  = load_predict(args)
    #end_read_data = time.time()
#    print "\nread done! totally cost: ",end_read_data - read_data,"\n"
#    print ("length of test_x: %d"% len(test_x))
#    print ("length of test_index: %d"% len(test_index))
    checkpoint_dir = args.model_save_path
    
    with tf.Session() as sess:
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('Restore model failed!')

        if not os.path.exists(args.result_path):
            os.mkdir(args.result_path)
                
        for num in range(args.start_mic_num, args.end_mic_num + 1):
            mrc_name = args.data_path + args.name_prefix + str(num) + ".mrc"
            if not os.path.exists(mrc_name):
                print("%s is not exist!" % mrc_name)
                continue
            print("processing mrc %s..." % mrc_name)
            mrc = mrcfile.open(mrc_name)
            output_name = args.result_path + args.name_prefix + str(num)  + '.box'
            output = open(output_name, 'w')
    
            # result = []
            x_step_num = (args.dim_x - args.boxsize) / args.scan_step
            y_step_num = (args.dim_y - args.boxsize) / args.scan_step
            for i in xrange(x_step_num):
                for j in xrange(y_step_num):
                    test_x = []
                    x = i*args.scan_step
                    y = j*args.scan_step
                    img = sub_img(mrc.data,x, y, args.boxsize)
                    print "img: " , img
                    stddev = np.std(img)
                    print("the stddev of image %d is %.5f" % (num, stddev))
                    if stddev <= args.min_std or stddev >= args.max_std:
                        continue

                    img = mapstd(img)
                    print "img: " , img
                    test_x.append(img)

                    rotate_map(img)      # rotate 90
                    test_x.append(img)
    
                    rotate_map(img)      # rotate 180
                    test_x.append(img)
    
                    rotate_map(img)      # rotate 270
                    test_x.append(img)
                    test_x = np.reshape(test_x,[4,args.boxsize,args.boxsize,1])

                    pred = sess.run(deepem.logits,feed_dict={deepem.X: test_x})
                    #pred = sess.run(tf.nn.sigmoid(deepem.logits),feed_dict={deepem.X: test_x})
                    print "pred is ", pred
                    avg = pred.mean()
             #       result.append([x,y,avg,stddev])
                    print num,".mrc x = ",x,", y = ",y,", avgpred = ", avg
                    if avg >= 0.5:
                        output.write(str(x)+'\t'+str(y)+'\t'+args.boxsize+'\t'+args.boxsize)
                        print num," ",x," ", y," ",args.boxsize," ",args.boxsize
                    
            mrc.close()
            output.close
 

        


if __name__ == '__main__':
    predict()

