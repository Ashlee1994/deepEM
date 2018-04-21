#!/gpfs/share/software/anaconda2/bin/python
# encoding: utf-8

import os
import pandas as pd
import numpy as np
#from args import Train_Args, Predict_Args
import time
import mrcfile

def mapstd(mrcData):
    avg = mrcData.mean()
    stddev = np.std(mrcData, ddof = 1)
    data = mrcData.copy()
    data = (data - avg) / stddev
    return data

def rotate_map(particle):
    '''
    This function is to rotate map 90, 180, 270 degree
    '''
    particle[:] = map(list,zip(*particle[::-1]))

def read_particles(mic_path, dim_x, dim_y, boxsize, name_prefix, box_path, start_num, end_num):
    particles = []
    for num in range(start_num, end_num + 1):
        boxX = []
        boxY = []
        mrc_name = mic_path + name_prefix + str(num) + ".mrc"
        box_name = box_path + name_prefix + str(num) + ".box"
        boxfile = open(box_name, 'r')
        for line in boxfile:
            col = line.split()
            x = int(col[0]) - 1
            y = int(col[1]) - 1
            if x < 0: x = 0
            if y < 0: y = 0
            if ( x  + boxsize > dim_x) or (y + boxsize ) > dim_y:
                continue
            boxX.append(x)
            boxY.append(y)
        boxfile.close()
        if len( boxX )  == 0:
            continue
        mrc = mrcfile.open(mrc_name)
        mrcstd = mapstd(mrc.data)
        particle = np.zeros((boxsize, boxsize))
        for ii in range(len(boxX)):
            index_x = boxX[ii]
            index_y = boxY[ii]
            for row in range(boxsize):
                particle[row,:] = mrcstd[0][index_y][index_x:index_x + boxsize]
                index_y += 1
            particles.append(particle)
            rotate_map(particle)
            particles.append(particle)
            rotate_map(particle)
            particles.append(particle)
            rotate_map(particle)
            particles.append(particle)
        mrc.close()
    print "num of particles", len(particles) / 4
    return particles
 

#def load_train(args):
def load_train(args):
    '''
    This part include load training parameters and training data
    '''
    time_start = time.time()
    #args = Train_Args()
    
    if not os.path.exists( args.mic_path) and os.path.exists(args.positive1_box_path ) and os.path.exists(args.negative1_box_path ) :
        print "Please make sure the mic path, positive1 path and negative1 path are exist!"
        exit -1
    
    #positive1 = read_particles(args.mic_path, args.dim_x, args.dim_y, args.boxsize, args.name_prefix, \
    #                            args.positive1_box_path, 30001,30050)

    positive1 = read_particles(args.mic_path, args.dim_x, args.dim_y, args.boxsize, args.name_prefix, \
                                args.positive1_box_path, args.positive1_mic_start_num, args.positive1_mic_end_num)
    negative1 = read_particles(args.mic_path, args.dim_x, args.dim_y, args.boxsize, args.name_prefix, \
                                args.negative1_box_path, args.negative1_mic_start_num, args.negative1_mic_end_num)
    if args.num_positive1 > len(positive1)/4:
        args.num_positive1 = len(positive1)/4
        print "positive1 only have ", len(positive1)/4, "particles, num_positive1 is set to ",len(positive1)/4
    if args.num_negative1 > len(negative1)/4:
        args.num_negative1 = len(negative1)/4
        print "negative1 only have ", len(negative1)/4, "particles, num_negative2 is set to ",len(negative1)/4

    # if do train again
    if args.do_train_again :
        if not os.path.exists( args.positive2_box_path) and os.path.exists( args.negative2_box_path):
            print "Please make sure the positive2 and negative2 path are exist!"
            exit -1
        
        positive2 = read_particles(args.mic_path, args.dim_x, args.dim_y, args.boxsize, args.name_prefix, \
                                 args.positive2_box_path, args.positive2_mic_start_num, args.positive2_mic_end_num)
        negative2 = read_particles(args.mic_path, args.dim_x, args.dim_y, args.boxsize, args.name_prefix, \
                                 args.negative2_box_path, args.negative2_mic_start_num, args.negative2_mic_end_num)
        if args.num_positive2 > len(positive2)/4:
            args.num_positive2 = len(positive2)/4
            print "positive2 only have ", len(positive2)/4, "particles, num_positive2 is set to ",len(positive2)/4
        if args.num_negative2 > len(negative2)/4:
            args.num_negative2 = len(negative2)/4
            print "negative2 only have ", len(negative2)/4, "particles, num_negative2 is set to ",len(negative2)/4
    else:
        args.num_positive2 = 0
        args.num_negative2 = 0

    print "positive1 len: ",np.array(positive1).shape
 
    train_size = (args.num_positive1 + args.num_negative1 + args.num_positive2 + args.num_negative2) * 4
    test_size = (args.num_p_test + args.num_n_test) * 4

    train_x = np.empty((train_size, args.boxsize, args.boxsize))
    train_y = np.zeros((train_size, 1))
    test_x = np.empty((test_size, args.boxsize, args.boxsize))
    test_y = np.zeros((test_size, 1))
    
    # todo: put data into train_x
    start = 0
    end = args.num_positive1*4
    train_x[start:end] = positive1[0:args.num_positive1*4]
    train_y[start:end] = 1

    start = end
    end += args.num_negative1*4
    train_x[start:end] = negative1[0:args.num_negative1*4]
    train_y[start:end] = 0

    if args.do_train_again: 
        start = end
        end += args.num_positive2*4
        train_x[start:end] =  positive2[0:args.num_positive2*4]
        train_y[start:end] = 1

        start = end
        end += args.num_negative2*4
        train_x[start:end] =  positive2[0:args.num_negative2*4]
        train_y[start:end] = 0
        
    # put data into test_x, test_y     

        if args.num_p_test > len(positive1)/4 + len(positive2)/4 - args.num_positive1 - args.num_positive2:
            args.num_p_test = len(positive1)/4+len(positive2)/4 - args.num_positive1 - args.num_positive2
            print "num_p_test is larger than the rest of positive particles, num_p_test is set to ",args.num_p_test
        if args.num_n_test > len(negative1)/4 + len(negative2)/4 - args.num_positive1 - args.num_positive2:
            args.num_n_test = len(negative1)/4 + len(negative2)/4 - args.num_positive1 - args.num_positive2
            print "num_n_test is larger than the rest of negative particles, num_n_test is set to ",args.num_n_test
        
        start = 0
        end = args.num_p_test*4
        if len(positive1) - args.num_positive1*4 >= args.num_p_test * 4:
            test_x[start:end] = positive1[args.num_positive1*4:args.num_positive1*4 + args.num_p_test*4]
        else:
            test_x[start : start + len(positive1)-args.num_positive1*4] = positive1[args.num_positive1*4:]
            test_x[start + len(positive1)-args.num_positive1*4:end] = positive2[args.num_positive2*4:args.num_p_test*4-len(positive1) + args.num_positive1*4]
        test_y[start:end] = 1
    
        start = end
        end += args.num_n_test*4
        if len(negative1) - args.num_negative1*4 >= args.num_n_test*4:
            test_x[start:end] = negative1[args.num_negative1*4:args.num_negative1*4 + args.num_n_test*4]
        else:
            test_x[start:start + len(negative1)-args.num_negative1*4] = negative1[args.num_negative1*4:]
            test_x[start+ len(positive1)-args.num_positive1*4:end] = positive2[args.num_positive2*4:args.num_p_test*4-len(positive1) + args.num_positive1*4]
        test_y[start:end] = 0
    else: # do_train_again = 0
        if args.num_p_test > len(positive1)/4 - args.num_positive1:
            args.num_p_test = len(positive1)/4 - args.num_positive1 
            print "num_p_test is larger than the rest of positive particles, num_p_test is set to ",args.num_p_test
        if args.num_n_test > len(negative1)/4 - args.num_positive1:
            args.num_n_test = len(negative1)/4 - args.num_positive1
            print "num_n_test is larger than the rest of negative particles, num_n_test is set to ",args.num_n_test

        start = 0
        end = args.num_p_test*4
        test_x[start:end] = positive1[args.num_positive1*4:args.num_positive1*4 + args.num_p_test*4]
        test_y[start:end] = 1

        start = end
        end += args.num_n_test*4
        test_x[start:end] = negative1[args.num_negative1*4:args.num_negative1*4 + args.num_n_test*4]
        test_y[start:end] = 0

    #train_x = train_data.astype('float32')
    #test_y = train_data.astype('float32')

    train_x = train_x.reshape(len(train_x),args.boxsize, args.boxsize, 1)
    test_x = test_x.reshape(len(test_x),args.boxsize, args.boxsize, 1)

    time_end = time.time()
    print "\nread done! totally cost: ",time_end - time_start,"\n"
    
    return train_x, train_y, test_x, test_y


def load_predict(args):
    '''
    This part include load predict parameters and predict data
    '''

if __name__ == '__main__':
    load_train()
