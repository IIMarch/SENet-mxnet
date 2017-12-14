import sys
sys.path.insert(0, '/data2/obj_detect/mxnet/0.11.0/python')
import mxnet as mx
import numpy as np
from memonger import memonger
from SE_ResNeXt_101 import get_symbol
#net = get_symbol(num_classes=1000, memonger=True, use_global_stats=True)
#network = memonger.search_plan(net, data=(120,3,224,224))

#sym, arg_params, aux_params = mx.model.load_checkpoint('./SE-ResNeXt-101', 0)
sym, arg_params, aux_params = mx.model.load_checkpoint('./caffe_converter/SE-ResNeXt-50', 0)


model = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=['prob_label'])
model.bind(data_shapes=[('data', (1, 3, 224, 224))], label_shapes=[('prob_label', (1,))])
#model = mx.mod.Module(symbol=network, context=mx.gpu(2), label_names=['softmax_label'])
#model.bind(data_shapes=[('data', (120, 3, 224, 224))], label_shapes=[('softmax_label', (120,))])
model.set_params(arg_params=arg_params, aux_params=aux_params)

input_data = np.load('./input_data.npy')
output_data = np.load('./prob_data.npy')

input_data[:, [0,2],:,:] = input_data[:, [2,0],:,:]
#input_data = mx.nd.array(np.random.rand(120,3,224,224))
batch = mx.io.DataBatch(data = [mx.nd.array(input_data)])



#model.forward(batch, is_train=True)
#model.backward()
model.forward(batch, is_train=False)
outputs = [output.asnumpy() for output in model.get_outputs()]
import ipdb; ipdb.set_trace()
print np.sum(np.abs(outputs[0] - output_data))






