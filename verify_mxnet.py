import sys
sys.path.insert(0, '/data2/obj_detect/mxnet/0.11.0/python')
import mxnet as mx
data = mx.symbol.Variable(name='data')
conv1_7x7_s2 = mx.symbol.Convolution(name='conv1_7x7_s2', data=data , num_filter=64, pad=(3, 3), kernel=(7,7), stride=(2,2), no_bias=True)
conv1_7x7_s2_bn = mx.symbol.BatchNorm(name='conv1_7x7_s2_bn', data=conv1_7x7_s2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv1_7x7_s2_bn_scale = conv1_7x7_s2_bn
conv1_relu_7x7_s2 = mx.symbol.Activation(name='conv1_relu_7x7_s2', data=conv1_7x7_s2_bn_scale , act_type='relu')
pool1_3x3_s2 = mx.symbol.Pooling(name='pool1_3x3_s2', data=conv1_relu_7x7_s2 , pooling_convention='full', pad=(0,0), kernel=(3,3), stride=(2,2), pool_type='max')
conv2_1_1x1_reduce = mx.symbol.Convolution(name='conv2_1_1x1_reduce', data=pool1_3x3_s2 , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv2_1_1x1_reduce_bn = mx.symbol.BatchNorm(name='conv2_1_1x1_reduce_bn', data=conv2_1_1x1_reduce , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv2_1_1x1_reduce_bn_scale = conv2_1_1x1_reduce_bn
conv2_1_1x1_reduce_relu = mx.symbol.Activation(name='conv2_1_1x1_reduce_relu', data=conv2_1_1x1_reduce_bn_scale , act_type='relu')
conv2_1_3x3 = mx.symbol.Convolution(name='conv2_1_3x3', data=conv2_1_1x1_reduce_relu , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True, num_group=32)
conv2_1_3x3_bn = mx.symbol.BatchNorm(name='conv2_1_3x3_bn', data=conv2_1_3x3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv2_1_3x3_bn_scale = conv2_1_3x3_bn
conv2_1_3x3_relu = mx.symbol.Activation(name='conv2_1_3x3_relu', data=conv2_1_3x3_bn_scale , act_type='relu')
conv2_1_1x1_increase = mx.symbol.Convolution(name='conv2_1_1x1_increase', data=conv2_1_3x3_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv2_1_1x1_increase_bn = mx.symbol.BatchNorm(name='conv2_1_1x1_increase_bn', data=conv2_1_1x1_increase , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv2_1_1x1_increase_bn_scale = conv2_1_1x1_increase_bn
conv2_1_global_pool = mx.symbol.Pooling(name='conv2_1_global_pool', data=conv2_1_1x1_increase_bn_scale , pooling_convention='full', global_pool=True, kernel=(1,1), pool_type='avg')
conv2_1_1x1_down = mx.symbol.Convolution(name='conv2_1_1x1_down', data=conv2_1_global_pool , num_filter=16, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv2_1_1x1_down_relu = mx.symbol.Activation(name='conv2_1_1x1_down_relu', data=conv2_1_1x1_down , act_type='relu')
conv2_1_1x1_up = mx.symbol.Convolution(name='conv2_1_1x1_up', data=conv2_1_1x1_down_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv2_1_prob = mx.symbol.Activation(name='conv2_1_prob', data=conv2_1_1x1_up , act_type='sigmoid')
conv2_1_1x1_proj = mx.symbol.Convolution(name='conv2_1_1x1_proj', data=pool1_3x3_s2 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv2_1_1x1_proj_bn = mx.symbol.BatchNorm(name='conv2_1_1x1_proj_bn', data=conv2_1_1x1_proj , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv2_1_1x1_proj_bn_scale = conv2_1_1x1_proj_bn
conv2_1 = mx.sym.broadcast_mul(conv2_1_1x1_up, conv2_1_1x1_increase) + conv2_1_1x1_proj
