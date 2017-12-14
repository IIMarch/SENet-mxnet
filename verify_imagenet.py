import sys
sys.path.insert(0, '/data2/obj_detect/mxnet/0.11.0/python')
import mxnet as mx
import numpy as np
from memonger import memonger
from SE_ResNeXt_101 import get_symbol
import cv2
from augmenter import Augmenter
aug = Augmenter((3,224,224), 256, 0, 0, 0, np.array([104,117,123]), np.array([1,1,1]))
#net = get_symbol(num_classes=1000, memonger=True, use_global_stats=True)
#network = memonger.search_plan(net, data=(120,3,224,224))

#sym, arg_params, aux_params = mx.model.load_checkpoint('./SE-ResNeXt-101', 0)
sym, arg_params, aux_params = mx.model.load_checkpoint('./SENet', 0)


model = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=['prob_label'])
model.bind(data_shapes=[('data', (1, 3, 224, 224))], label_shapes=[('prob_label', (1,))])
#model = mx.mod.Module(symbol=network, context=mx.gpu(2), label_names=['softmax_label'])
#model.bind(data_shapes=[('data', (120, 3, 224, 224))], label_shapes=[('softmax_label', (120,))])
model.set_params(arg_params=arg_params, aux_params=aux_params)

import ipdb; ipdb.set_trace()
input_data = cv2.imread('./ILSVRC2012_val_00011111.JPEG')
input_data = aug(input_data)
input_data = input_data.transpose(2,0,1)[np.newaxis, :,:,:]
input_data = input_data[:,::-1,:,:]

#input_data = mx.nd.array(np.random.rand(120,3,224,224))
batch = mx.io.DataBatch(data = [mx.nd.array(input_data)])



#model.forward(batch, is_train=True)
#model.backward()
model.forward(batch, is_train=False)
outputs = [output.asnumpy() for output in model.get_outputs()]
print outputs[0].max()
import ipdb; ipdb.set_trace()






