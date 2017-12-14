import sys
sys.path.insert(0, '/data2/obj_detect/mxnet/0.11.0/python')
import mxnet as mx
import numpy as np

def get_symbol():
    data = mx.sym.var(name='data') # NCHW
    scale = mx.sym.var(name='scale') # NC11
    net = mx.sym.broadcast_mul(data, scale)
    return net

def get_input_shape():
    return {'data':(2,3,2,2), 'scale':(2,3,1,1)}

def run():
    net = get_symbol()
    input_shape = get_input_shape()
    exec_ = net.simple_bind(ctx=mx.cpu(), **input_shape)

    data = np.random.rand(2,3,2,2)
    data = mx.nd.array(data)

    scale = np.random.rand(2,3,1,1)
    scale = mx.nd.array(scale)

    data.copyto(exec_.arg_dict['data'])
    scale.copyto(exec_.arg_dict['scale'])

    exec_.forward(is_train=False)

    outputs = [output.asnumpy() for output in exec_._get_outputs()]

    import ipdb; ipdb.set_trace()
    asd = 0;

run()
