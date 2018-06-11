import os
import mrcfile
import numpy as np

from utils import mapstd
from PIL import Image
import cv2 as cv
from utils import load_train
from model import deepEM
from args_KLH import Train_Args,Predict_Args
output_name = "output.txt"
output = open(output_name, 'w')
boxsize = 160

#mrcname = "data/KLHdata/mic/14.mrc"
mrcname = "data/19Sdata/mrc_file/image_30008.mrc"

mrc_x = 1855
mrc_y = 1919
# 69    624
x = 69
y = 624
y = mrc_y - y - boxsize

#y -= boxsize
#x = 100
#y = 0


#x =int( x - boxsize/2)
#y = int(y - boxsize/2)
if x < 0 :
    x = 0
if y < 0:
    y = 0
if x > mrc_x:
    x = mrc_x  
if y > mrc_y:
    y = mrc_y
mrc = mrcfile.open(mrcname)
print("mrc.data shape: ",mrc.data.shape)
print("mrc.data: ", mrc.data)
mrcstd = mapstd(mrc.data, mrc_x, mrc_y)

print("mrc shape: ",mrc.data.shape)
print("mrcstd: ",mrcstd)
#output.write("origin: map")
#output.write(str(mrcstd))
print("mrc shape: ", mrcstd.shape)

print("type mrcstd:",type(mrcstd))
#micrographDataInt = (255.0 * mrcstd).astype(np.uint8)
#data = Image.fromarray(micrographDataInt.reshape(mrc_x,mrc_x))

#cv.imshow("mrc: ",data)
#cv.waitKey(0)

#mrcstd = np.reshape(mrcstd,[1855,1919])
#print("mrc shape", mrcstd.shape)

particle = np.zeros((boxsize, boxsize))

'''
for row in range(boxsize):
    particle[row,:] = mrcstd[0][y][x:x + boxsize]
    y += 1
'''
#mrcstd = np.reshape(mrcstd,[2048, 2048])
#temImage = cv.resize(mrcstd,(1000,1000),interpolation=cv.INTER_CUBIC)
 
#cv.imshow("mrcstd",temImage)
#cv.waitKey(0) 

#print("mrcstd shape: ", mrcstd.shape)
#rot90 = np.rot90(mrcstd)
#cv.imshow("particle: ",rot90)
#cv.waitKey(0) 

#exam = np.reshape(mrcstd,[-1])
#print("exam shape: ", exam.shape)
#exam = np.reshape(exam,[2048,2048])

#print("exam shape: ", exam.shape)


#part = exam[mrc_x*x+y:mrc_x*x+y+boxsize*boxsize]

for row in range(boxsize):
    #for col in range(boxsize):
    particle[row,:] = mrcstd[y][x:x+boxsize]
    #  print("x = %d, y = %d" %(x,y))
    #particle[row,:] = mrc.data[0][y][x:x+boxsize]
    #particle[row,:] = mrcstd[x][y:y+bo]
    y += 1


'''
for row in range(boxsize):
    particle[row,:] = mrcstd[x][y:y+boxsize]
    #print("x = %d, y = %d" %(x,y))
    x -= 1
'''
#args = Train_Args()
#train_x, train_y, test_x, test_y = load_train(args)

#img = np.reshape(mrcstd, [])
#print("particle shape", particle.shape)
#particel = np.rot90(particle)
#particle = mapstd(particle)
#output.write("particle: x = " + str(x) +", y = " + str(y))
#img = np.reshape(particle,[boxsize,boxsize])

micrographDataInt = (255.0 * particle).astype(np.uint8)

image = Image.fromarray(micrographDataInt)
image_resize = image.resize((224,224),Image.ANTIALIAS)
image_resize = np.array(image_resize)

cv.imshow("particle: ",micrographDataInt)
cv.waitKey(0)  
cv.imshow("particle: ",image_resize )
cv.waitKey(0)  
output.write(str(particle))
