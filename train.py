import tensorflow as tf
import numpy as np
import os

from utils import load_train
from model import deepEM
from args import Train_Args,Predict_Args


def train():
    args = Train_Args()
    train_x, train_y, test_x, test_y = load_train(args)
    checkpoint_dir = "./save"
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
#    gpu_config = tf.ConfigProto()
#    gpu_config.gpu_options.allow_growth = True
#    with tf.Session(config = gpu_config) as sess:
    with tf.Session() as sess:
        deepem = deepEM(args)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep = 100)
        sess.run(tf.global_variables_initializer())
        print("train size is %d " % len(train_x))
        for e in xrange(args.num_epochs):
            print('\n=============== Batch %d/%d ==============='% (e,args.num_epochs + 1))
            cost = []
            num_batch = len(train_x) / args.batch_size
            print("num_batch is %d" % num_batch)
            for i in xrange(num_batch):
                batch_x = train_x[args.batch_size*i:args.batch_size*(i+1)]
                batch_y = train_y[args.batch_size*i:args.batch_size*(i+1)]
                batch_x = np.asarray(batch_x)
                batch_y = np.asarray(batch_y)
                feed_dict = {deepem.X:batch_x, deepem.Y: batch_y}
                fetches = [deepem.cost_func, deepem.optimizer]

                loss,_= sess.run(fetches, feed_dict)
                cost.append(loss)
                if i % 100 == 0:
                    #print('i =  %d'% i)
                    print('Loss: %.6f' % (np.mean(cost)))
            #print "layer 7 input: ", deepem.l7_input.eval()
            ckpt_path = os.path.join(checkpoint_dir, 'model.ckpt')
            saver.save(sess, ckpt_path, global_step = e)

        # test
        saver.restore(sess, checkpoint_dir)
        test_pred = tf.argmax(deepem.logits,1).ecal({deepem.X: test_x})
        test_true = np.argmax(test_x, 1)
        test_correct = correct.eval({deepem.X: test_x, deepem.Y: test_y})
        test_accuracy = np.sum(test_correct)/len(test_correct)
        print('test set accuracy: ', test_accuracy)


if __name__ == '__main__':
    train()

