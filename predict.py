
import numpy as np
from PIL import Image

import os
os.environ["GLOG_minloglevel"] = "1"
import caffe


caffe.set_mode_cpu()

# read labels
with open('words1000.txt', 'r') as f:
	words=list(f)


# load model
net = caffe.Net('deploy.prototxt', 1, weights='resnet50_cvgj_iter_112099.caffemodel')
print '--------------------------------------------------------------------------------'

# read image
im = np.array(Image.open('input3.JPEG'))

#crop image
im2 = caffe.io.oversample((im,),(224,224))

#take first crop
net.blobs['data'].data[...] = np.transpose(im2[0:1,:,:,:], (0,3,1,2))

# forward propagation
net.forward()

#compute output
softmax_probabilities = net.blobs['predict'].data[0,:]
#compute top 5 sorted in ascending probability order
ranks = np.argpartition(softmax_probabilities, -5)[-5:]
ranks = ranks[np.argsort(softmax_probabilities[ranks])]
# print top 5
print ranks
for rank in ranks:
	print "output rank {} - proba : {:05.2f} % - label  {} ".format(rank,100*softmax_probabilities[rank],words[rank].rstrip())


#print("softmax: ",softmax_probabilities)
