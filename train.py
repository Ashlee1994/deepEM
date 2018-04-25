import tensorflow as tf
import numpy as np
import time
import os

from utils import load_train
from model import deepEM
from KLH import Train_Args,Predict_Args


def train():
    args = Train_Args()
    train_start = time.time()
    time_start = time.time()

    checkpoint_dir = args.model_save_path
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
        
    output_name = args.model_save_path + "training_messages.txt"
    output = open(output_name, 'w')
    print("read data start.")
    output.write("read data start.\n")

    train_x, train_y, test_x, test_y = load_train(args)
    time_end = time.time()
    print("\nread done! totally cost: %.5f \n" %(time_end - time_start))
    output.write("read done! totally cost: " + str(time_end - time_start) +'\n')
    print("training start.")

    time_start = time.time()
    
#    gpu_config = tf.ConfigProto()
#    gpu_config.gpu_options.allow_growth = True
#    with tf.Session(config = gpu_config) as sess:



    with tf.Session() as sess:
        deepem = deepEM(args)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep = 100)
        sess.run(tf.global_variables_initializer())
        print("train size is %d " % len(train_x))
        for e in range(args.num_epochs):
            print('\n=============== Batch %d/%d ==============='% (e + 1,args.num_epochs))
            output.write("\n=============== Batch " + str(e + 1) + str(args.num_epochs) + " ===============\n")
            cost = []
            num_batch = len(train_x) // args.batch_size
            print("num_batch is %d" % num_batch)
            for i in range(num_batch):
                batch_x = train_x[args.batch_size*i:args.batch_size*(i+1)]
                batch_y = train_y[args.batch_size*i:args.batch_size*(i+1)]
                batch_x = np.asarray(batch_x)
                batch_y = np.asarray(batch_y)
                # sheffle training data
                #train_x_batch, train_y_batch = tf.train.shuffle_batch([train_x,train_y], batch_size = args.batch_size,capacity = 100,min_after_dequeue = 50)
                #feed_dict = {deepem.X:batch_x_batch, deepem.Y: batch_y_batch}
                feed_dict = {deepem.X:batch_x, deepem.Y: batch_y}
                fetches = [deepem.loss, deepem.optimizer]

                loss,_= sess.run(fetches, feed_dict)
                cost.append(loss)
                if i % 3 == 0:
                    #print('i =  %d'% i)
                    print('Loss: %.6f' % (np.mean(cost)))
                    output.write("loss: " + str(np.mean(cost)) + '\n')
            #print "layer 7 input: ", deepem.l7_input.eval()
            #train_accuracy = accuracy.eval( feed_dict )
            #print("batch %d, training accuracy %.6f" %(e, train_accuracy))
            ckpt_path = os.path.join(checkpoint_dir, 'model.ckpt')
            saver.save(sess, ckpt_path, global_step = e)

        # test
        checkpoint_dir = args.model_save_path
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('Restore model failed!')
        
        time_end = time.time()
        print("\ntraining done! totally cost: %.5f \n" %(time_end - time_start))
        output.write("training done! totally cost: " + str(time_end - time_start) + '\n')
        print("testing start.")

        time_start = time.time()
        test_pred = sess.run(tf.nn.sigmoid(deepem.logits), feed_dict={deepem.X: test_x})
        #print "pred is ", test_pred
        #print "test_y is ", test_y
        test_pred[test_pred<0.5] = 0
        test_pred[test_pred>=0.5] = 1
        #print "pred is ", test_pred
        accuracy = np.equal(test_pred,test_y)
        print("accuracy: %.6f" % np.sum(accuracy)/len(accuracy))
        output.write("accuracy:" + str(np.sum(accuracy)/len(accuracy)) + '\n')
        time_end = time.time()
        print("\ntesting done! totally cost: %.5f \n" %(time_end - time_start))
        output.write("testing done! totally cost: " + str(time_end - time_start) + '\n')
    train_end = time.time()
    print("\ntrain done! totally cost: %.5f \n" %(train_end - train_start))
    output.write("testing done! totally cost: " + str(train_end - train_start) + '\n')
    output.close


if __name__ == '__main__':
    train()

