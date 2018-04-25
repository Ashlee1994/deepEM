import tensorflow as tf
import numpy as np
import mrcfile
import time
import os

from utils import load_predict
from utils import sub_img
from utils import mapstd
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
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    with tf.Session(config = gpu_config) as sess:
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
            count = 0 
            x_step_num = (args.dim_x - args.boxsize) // args.scan_step
            y_step_num = (args.dim_y - args.boxsize) // args.scan_step
            for i in range(x_step_num):
                for j in range(y_step_num):
                    test_x = []
                    x = i*args.scan_step
                    y = j*args.scan_step
                    y_ = y
                    y = int(args.dim_y -y - args.boxsize)
                    img = sub_img(mrc.data,x, y, args.boxsize)
              #      print("img: " , img)
                    stddev = np.std(img)
               #     print("the stddev of image %d is %.5f" % (num, stddev))
                    if stddev <= args.min_std or stddev >= args.max_std:
                        continue

                    img = mapstd(img)
               #     print("img: " , img)
                    test_x.append(img)

                    img = np.rot90(img)
                    test_x.append(img)

                    img = np.rot90(img)
                    test_x.append(img)

                    img = np.rot90(img)
                    test_x.append(img)

                   # print("test_x: " , test_x)

                #    print("img.shape: ",img.shape)


                    test_x = np.asarray(test_x).reshape(4,args.boxsize,args.boxsize,1)

                    #pred = sess.run(deepem.logits,feed_dict={deepem.X: test_x})
                    pred = sess.run(tf.nn.softmax(deepem.logits),feed_dict={deepem.X: test_x})
                  #  print("pred is %s", pred)
                    #avg = pred.mean()
              #      print("avg is %s" % avg)
             #       result.append([x,y,avg,stddev])
                    prob = np.mean(pred,0)
                    print("%d .mrc x = %d, y = %d, pre = %.5f" %(num,x,y_,prob[0]))
                    if prob[0] > 0.9999:
                        #print("%d .mrc x = %d, y = %d, pre = %s, avgpred = %s" %(num,x,y,pred[0], avg))
                        output.write(str(x)+'\t'+str(y_)+'\t'+str(args.boxsize)+'\t'+str(args.boxsize) +'\n')
                        print(num,"  ",x,"  ",y,"  ",prob[0])
                        count += 1
                    
            print("%d particles in this mrc!" % count)
            mrc.close()
            output.close
 

        


if __name__ == '__main__':
    predict()

