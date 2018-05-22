# encoding: utf-8

import os,random
import pandas as pd
import numpy as np
import mrcfile
import cv2 as cv
from PIL import Image

def mapstd(mrcData,dim_x,dim_y):
    avg = mrcData.mean()
    stddev = np.std(mrcData,ddof=1)
    minval = mrcData.min()
    maxval = mrcData.max()
    data = mrcData.copy()
    sigma_contrast = 0.3
    if minval == maxval or sigma_contrast > 0.:
        minval = avg - sigma_contrast * stddev
        maxval = avg + sigma_contrast * stddev
    if sigma_contrast > 0 or minval != maxval:
        data[data < minval] = minval
        data[data > maxval] = maxval
    if data.max == 0 and data.min == 0:
        normalize = 0.0
    else:
        data = ( data - data.min())/(data.max() - data.min())
    # 19S data
    data = data[::-1]
    data = data.reshape(dim_y,dim_x)
    return data

def sub_img(mrcData, x,y,boxsize):
    '''
    This part is to extract a box of pixels form origin mrc file
    '''
    box = np.empty((boxsize,boxsize))
    for row in range(boxsize):
        box[row,:] = mrcData[y][x:x+boxsize]
        y += 1
    return box

def read_particles(mic_path, dim_x, dim_y, boxsize, name_prefix, box_path, start_num, end_num,name_length,rotation_angel,rotation_n):
    particles = []
    for num in range(start_num, end_num + 1):
        boxX = []
        boxY = []
        mrc_name = mic_path + name_prefix + str(num).zfill(name_length) + ".mrc"
        box_name = box_path + name_prefix + str(num).zfill(name_length) + ".box"
        if not os.path.exists(mrc_name):
            print("%s is not exist!" % mrc_name)
            continue
        if not os.path.exists(box_name):
            print("%s is not exist!" % box_name)
            continue
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
        mrcstd = mapstd(mrc.data,dim_x,dim_y)
        particle = np.zeros((boxsize, boxsize))
        for ii in range(len(boxX)):
            index_x = boxX[ii]
            index_y = dim_y - boxY[ii] -boxsize
            particle = sub_img(mrcstd,index_x, index_y, boxsize)
            # rotate map
            image = Image.fromarray(particle)
            for i in range(rotation_n):
                image_rot = image.rotate(rotation_angel*i)
                particles.append(np.array(image_rot))
        mrc.close()
    return particles

def load_train(args):
    '''
    This part include load training parameters and training data
    '''
    if not os.path.exists( args.mic_path) and os.path.exists(args.positive1_box_path ) and os.path.exists(args.negative1_box_path ) :
        print("Please make sure the mic path, positive1 path and negative1 path are exist!")
        exit -1

    positive1 = read_particles(args.mic_path, args.dim_x, args.dim_y, args.boxsize, args.name_prefix, \
                                args.positive1_box_path, args.positive1_mic_start_num, args.positive1_mic_end_num, \
                                args.name_length,args.rotation_angel,args.rotation_n)
    negative1 = read_particles(args.mic_path, args.dim_x, args.dim_y, args.boxsize, args.name_prefix, \
                                args.negative1_box_path, args.negative1_mic_start_num, args.negative1_mic_end_num, \
                                args.name_length,args.rotation_angel,args.rotation_n)

    # shuffle positive1 and negative1
    random.shuffle(positive1)
    random.shuffle(negative1)

    positive1_particles_num = len(positive1)//args.rotation_n
    negative1_particles_num = len(negative1)//args.rotation_n

    print("positive1 size:" , positive1_particles_num)
    print("negative1 size:" , negative1_particles_num)
    if args.num_positive1 > positive1_particles_num:
        args.num_positive1 = positive1_particles_num
        print("positive1 only have %d particles, num_positive1 is set to %d" % (positive1_particles_num,positive1_particles_num))
    if args.num_negative1 > negative1_particles_num:
        args.num_negative1 = negative1_particles_num
        print("negative1 only have %d particles, num_negative2 is set to %d" % (negative1_particles_num, negative1_particles_num))

    # if do train again
    if args.do_train_again :
        if not os.path.exists( args.positive2_box_path) and os.path.exists( args.negative2_box_path):
            print("Please make sure the positive2 and negative2 path are exist!")
            exit -1
        
        positive2 = read_particles(args.mic_path, args.dim_x, args.dim_y, args.boxsize, args.name_prefix, \
                                 args.positive2_box_path, args.positive2_mic_start_num, args.positive2_mic_end_num, \
                                 args.name_length,args.rotation_angel,args.rotation_n)
        negative2 = read_particles(args.mic_path, args.dim_x, args.dim_y, args.boxsize, args.name_prefix, \
                                 args.negative2_box_path, args.negative2_mic_start_num, args.negative2_mic_end_num, \
                                 args.name_length,args.rotation_angel,args.rotation_n)

        # shuffle positive2 and negative2
        random.shuffle(positive2)
        random.shuffle(negative2)

        positive2_particles_num = len(positive2)//args.rotation_n
        negative2_particles_num = len(negative2)//args.rotation_n

        print("positive2 size:" , positive2_particles_num)
        print("negative2 size:" , negative2_particles_num)

        if args.num_positive2 > positive2_particles_num:
            args.num_positive2 = positive2_particles_num
            print("positive2 only have %d particles, num_positive2 is set to %d" %(positive2_particles_num, positive2_particles_num))
        if args.num_negative2 > negative2_particles_num:
            args.num_negative2 = negative2_particles_num
            print("negative2 only have %d particles, num_negative2 is set to %d" % (negative2_particles_num, negative2_particles_num))
    else:
        args.num_positive2 = 0
        args.num_negative2 = 0

    print("positive1 len: " , np.array(positive1).shape)
 
    train_size = int((args.num_positive1 + args.num_negative1 + args.num_positive2 + args.num_negative2) * args.rotation_n)
    test_size = int((args.num_p_test + args.num_n_test) * args.rotation_n)

    train_x = np.empty((train_size, args.boxsize, args.boxsize))
    train_y = np.zeros((train_size, 1))
    test_x = np.empty((test_size, args.boxsize, args.boxsize))
    test_y = np.zeros((test_size, 1))

    start = 0
    end = args.num_positive1*args.rotation_n
    train_x[start:end] = positive1[0:args.num_positive1*args.rotation_n]
    train_y[start:end] = 1

    start = end
    end += args.num_negative1*args.rotation_n
    train_x[start:end] = negative1[0:args.num_negative1*args.rotation_n]
    train_y[start:end] = 0

    if args.do_train_again: 
        start = end
        end += args.num_positive2*args.rotation_n
        train_x[start:end] =  positive2[0:args.num_positive2*args.rotation_n]
        train_y[start:end] = 1

        start = end
        end += args.num_negative2*args.rotation_n
        train_x[start:end] =  negative2[0:args.num_negative2*args.rotation_n]
        train_y[start:end] = 0
        
        # put data into test_x, test_y     

        if args.num_p_test > positive1_particles_num + positive2_particles_num - args.num_positive1 - args.num_positive2:
            args.num_p_test = positive1_particles_num + positive2_particles_num - args.num_positive1 - args.num_positive2
            print("num_p_test is larger than the rest of positive particles, num_p_test is set to %d" % args.num_p_test)
        if args.num_n_test > negative1_particles_num + negative2_particles_num - args.num_positive1 - args.num_positive2:
            args.num_n_test = negative1_particles_num + negative2_particles_num - args.num_positive1 - args.num_positive2
            print ("num_n_test is larger than the rest of negative particles, num_n_test is set to %d" % args.num_n_test)
        
        start = 0
        end = args.num_p_test*args.rotation_n
        if len(positive1) - args.num_positive1*args.rotation_n >= args.num_p_test * args.rotation_n:
            test_x[start:end] = positive1[args.num_positive1*args.rotation_n:(args.num_positive1 + args.num_p_test)*args.rotation_n]
        else:
            test_x[start : start + len(positive1)-args.num_positive1*args.rotation_n] = positive1[args.num_positive1*args.rotation_n:]
            test_x[start + len(positive1)-args.num_positive1*args.rotation_n:end] = \
                positive2[args.num_positive2*args.rotation_n:(args.num_positive2+args.num_p_test+ args.num_positive1)*args.rotation_n-len(positive1)]
        test_y[start:end] = 1

        start = end
        end += args.num_n_test*args.rotation_n
        if len(negative1) - args.num_negative1*args.rotation_n >= args.num_n_test*args.rotation_n:
            test_x[start:end] = negative1[args.num_negative1*args.rotation_n:(args.num_negative1 + args.num_n_test)*args.rotation_n]
        else:
            test_x[start:start + len(negative1)-args.num_negative1*args.rotation_n] = negative1[args.num_negative1*args.rotation_n:]
            test_x[start+ len(negative1)-args.num_negative1*args.rotation_n:end] =  \
                negative2[args.num_negative2*args.rotation_n: (args.num_negative2+args.num_n_test+ args.num_negative1)*args.rotation_n-len(negative1)]
        test_y[start:end] = 0
    else:  # do_train_again = 0
        if args.num_p_test > positive1_particles_num - args.num_positive1:
            args.num_p_test = positive1_particles_num - args.num_positive1 
            print("num_p_test is larger than the rest of positive particles, num_p_test is set to %d" % args.num_p_test)
        if args.num_n_test > negative1_particles_num - args.num_positive1:
            args.num_n_test = negative1_particles_num - args.num_positive1
            print ("num_n_test is larger than the rest of negative particles, num_n_test is set to %d" % args.num_n_test)

        start = 0
        end = args.num_p_test*args.rotation_n
        test_x[start:end] = positive1[args.num_positive1*args.rotation_n:(args.num_positive1 + args.num_p_test)*args.rotation_n]
        test_y[start:end] = 1

        start = end
        end += args.num_n_test*args.rotation_n
        test_x[start:end] = negative1[args.num_negative1*args.rotation_n:(args.num_negative1 + args.num_n_test)*args.rotation_n]
        test_y[start:end] = 0

    index = [i for i in range(len(train_x))]
    np.random.shuffle(index)
    train_x = train_x[index]
    train_y = train_y[index]

    train_x = train_x.reshape(len(train_x),args.boxsize, args.boxsize, 1)
    test_x = test_x.reshape(len(test_x),args.boxsize, args.boxsize, 1)
   
    return train_x, train_y, test_x, test_y

def non_max_suppression(particle,scores, boxsize, overlapThresh): 
    ## todo : sort in scores, and remove the overlapping particles.   
    """Pure Python NMS baseline."""    
    x1 = particle[:,0]   
    y1 = particle[:,1]  
    x2 = x1 + boxsize
    y2 = y1 + boxsize
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)    
    # sort socres, get the highest index    
    # order = scores.argsort()[::-1] 
    #idxs = np.argsort(y2) 
    #print("idxs is :" , idxs)  
    # keep is the result   
    pick = []    
    idxs = scores.copy()  #.sort()
    print("idxs is: ", idxs)
    obj = np.sort(idxs)
    print("idxs is: ", idxs)
    print("obj is: ", obj)
    print("sort: ", np.sort(idxs))
    while len(idxs) > 0:    
        # idxs[0] is the highest box
        last = len(idxs) - 1
        i = idxs[last]    
        # pick.append(i) 
        # calculate the overlapping area  
        xx1 = np.maximum(x1[i], x1[idxs[:last]])    
        yy1 = np.maximum(y1[i], y1[idxs[:last]])    
        xx2 = np.minimum(x2[i], x2[idxs[:last]])    
        yy2 = np.minimum(y2[i], y2[idxs[:last]])    
    
        w = np.maximum(0.0, xx2 - xx1 + 1)    
        h = np.maximum(0.0, yy2 - yy1 + 1)    

        overlap = (w * h) / areas[idxs[:last]] 

        # overlap > overlapThresh   select the maximun socre
        #highest = np.argmax(np.concatenate((scores[last],scores.where(overlap > overlapThresh)[0])))
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
        pick.append(highest)
    # print("keep:", keep)
    return particle[pick].astype("int")

# remove overlapping particles
def non_max_suppression_fast(particle,boxsize, overlapThresh):
    # if there are no boxes, return an empty list
    if len(particle) == 0:
        return []
 
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if particle.dtype.kind == "i":
        particle = particle.astype("float")
 
    # initialize the list of picked indexes 
    pick = []
 
    # grab the coordinates of the bounding boxes
    x1 = particle[:,0]   
    y1 = particle[:,1]  
    x2 = x1 + boxsize
    y2 = y1 + boxsize
 
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
 
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
 
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
 
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        #print("overlap is", overlap)
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
 
    # return only the bounding boxes that were picked using the
    # integer data type
    return particle[pick].astype("int")

def load_predict(args, mrc_name):
    '''
    This part include load predict parameters and predict data
    '''
    test_x = []
    test_index = []
    mrc = mrcfile.open(mrc_name)
    mrc_std = mapstd(mrc.data,args.dim_x,args.dim_y)
    
    x_step_num = (args.dim_x - args.boxsize) // args.scan_step
    y_step_num = (args.dim_y - args.boxsize) // args.scan_step
    for i in range(x_step_num):
        for j in range(y_step_num):
            x = i*args.scan_step
            y = j*args.scan_step       
            img = sub_img(mrc_std,x, y, args.boxsize)
            stddev = np.std(img)
            test_x.append(img)
            test_index.append([x,args.dim_y - args.boxsize - y])
    mrc.close()
        
    return test_x, test_index

if __name__ == '__main__':
    load_train()



















