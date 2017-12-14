import sys
sys.path.insert(0, '/data2/obj_detect/caffe/1.0.0/python')
import caffe
caffe.set_mode_gpu()
caffe.set_device(3)
import numpy as np
import time
np.random.seed(int(time.time()))
#net = caffe.Net('./SE-ResNeXt-101.prototxt', './SE-ResNeXt-101.caffemodel', caffe.TEST)
#net = caffe.Net('./SE-ResNet-50.prototxt', './SE-ResNet-50.caffemodel', caffe.TEST)
#net = caffe.Net('./SE-ResNet-101.prototxt', './SE-ResNet-101.caffemodel', caffe.TEST)
#net = caffe.Net('./SE-BN-Inception.prototxt', './SE-BN-Inception.caffemodel', caffe.TEST)
net = caffe.Net('./SE-ResNeXt-50.prototxt', './SE-ResNeXt-50.caffemodel', caffe.TEST)
input_data = np.random.rand(1,3,224,224)
net.blobs['data'].data[:] = input_data

np.save('input_data.npy', input_data)
net.forward()
for data in net.blobs:
    if data == 'prob':

        np_data = net.blobs[data].data
        np.save('{}_data.npy'.format(data), np_data)


