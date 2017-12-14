import mxnet as mx
data = mx.symbol.Variable(name='data')
conv1_7x7_s2 = mx.symbol.Convolution(name='conv1_7x7_s2', data=data , num_filter=64, pad=(3, 3), kernel=(7,7), stride=(2,2), no_bias=True)
conv1_7x7_s2_bn = mx.symbol.BatchNorm(name='conv1_7x7_s2_bn', data=conv1_7x7_s2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv1_7x7_s2_bn_scale = conv1_7x7_s2_bn
conv1_relu_7x7_s2 = mx.symbol.Activation(name='conv1_relu_7x7_s2', data=conv1_7x7_s2_bn_scale , act_type='relu')
pool1_3x3_s2 = mx.symbol.Pooling(name='pool1_3x3_s2', data=conv1_relu_7x7_s2 , pooling_convention='full', pad=(0,0), kernel=(3,3), stride=(2,2), pool_type='max')
conv2_1_1x1_reduce = mx.symbol.Convolution(name='conv2_1_1x1_reduce', data=pool1_3x3_s2 , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv2_1_1x1_reduce_bn = mx.symbol.BatchNorm(name='conv2_1_1x1_reduce_bn', data=conv2_1_1x1_reduce , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv2_1_1x1_reduce_bn_scale = conv2_1_1x1_reduce_bn
conv2_1_1x1_reduce_relu = mx.symbol.Activation(name='conv2_1_1x1_reduce_relu', data=conv2_1_1x1_reduce_bn_scale , act_type='relu')
conv2_1_3x3 = mx.symbol.Convolution(name='conv2_1_3x3', data=conv2_1_1x1_reduce_relu , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
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
if memonger:
    conv2_1_1x1_proj_bn_scale._set_attr(mirror_stage='True')
conv2_1 = mx.sym.broadcast_mul(conv2_1_prob, conv2_1_1x1_increase_bn_scale) + conv2_1_1x1_proj_bn_scale
conv2_1_relu = mx.symbol.Activation(name='conv2_1_relu', data=conv2_1 , act_type='relu')
conv2_2_1x1_reduce = mx.symbol.Convolution(name='conv2_2_1x1_reduce', data=conv2_1_relu , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv2_2_1x1_reduce_bn = mx.symbol.BatchNorm(name='conv2_2_1x1_reduce_bn', data=conv2_2_1x1_reduce , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv2_2_1x1_reduce_bn_scale = conv2_2_1x1_reduce_bn
conv2_2_1x1_reduce_relu = mx.symbol.Activation(name='conv2_2_1x1_reduce_relu', data=conv2_2_1x1_reduce_bn_scale , act_type='relu')
conv2_2_3x3 = mx.symbol.Convolution(name='conv2_2_3x3', data=conv2_2_1x1_reduce_relu , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
conv2_2_3x3_bn = mx.symbol.BatchNorm(name='conv2_2_3x3_bn', data=conv2_2_3x3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv2_2_3x3_bn_scale = conv2_2_3x3_bn
conv2_2_3x3_relu = mx.symbol.Activation(name='conv2_2_3x3_relu', data=conv2_2_3x3_bn_scale , act_type='relu')
conv2_2_1x1_increase = mx.symbol.Convolution(name='conv2_2_1x1_increase', data=conv2_2_3x3_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv2_2_1x1_increase_bn = mx.symbol.BatchNorm(name='conv2_2_1x1_increase_bn', data=conv2_2_1x1_increase , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv2_2_1x1_increase_bn_scale = conv2_2_1x1_increase_bn
conv2_2_global_pool = mx.symbol.Pooling(name='conv2_2_global_pool', data=conv2_2_1x1_increase_bn_scale , pooling_convention='full', global_pool=True, kernel=(1,1), pool_type='avg')
conv2_2_1x1_down = mx.symbol.Convolution(name='conv2_2_1x1_down', data=conv2_2_global_pool , num_filter=16, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv2_2_1x1_down_relu = mx.symbol.Activation(name='conv2_2_1x1_down_relu', data=conv2_2_1x1_down , act_type='relu')
conv2_2_1x1_up = mx.symbol.Convolution(name='conv2_2_1x1_up', data=conv2_2_1x1_down_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv2_2_prob = mx.symbol.Activation(name='conv2_2_prob', data=conv2_2_1x1_up , act_type='sigmoid')
if memonger:
    conv2_1_relu._set_attr(mirror_stage='True')
conv2_2 = mx.sym.broadcast_mul(conv2_2_prob, conv2_2_1x1_increase_bn_scale) + conv2_1_relu
conv2_2_relu = mx.symbol.Activation(name='conv2_2_relu', data=conv2_2 , act_type='relu')
conv2_3_1x1_reduce = mx.symbol.Convolution(name='conv2_3_1x1_reduce', data=conv2_2_relu , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv2_3_1x1_reduce_bn = mx.symbol.BatchNorm(name='conv2_3_1x1_reduce_bn', data=conv2_3_1x1_reduce , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv2_3_1x1_reduce_bn_scale = conv2_3_1x1_reduce_bn
conv2_3_1x1_reduce_relu = mx.symbol.Activation(name='conv2_3_1x1_reduce_relu', data=conv2_3_1x1_reduce_bn_scale , act_type='relu')
conv2_3_3x3 = mx.symbol.Convolution(name='conv2_3_3x3', data=conv2_3_1x1_reduce_relu , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
conv2_3_3x3_bn = mx.symbol.BatchNorm(name='conv2_3_3x3_bn', data=conv2_3_3x3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv2_3_3x3_bn_scale = conv2_3_3x3_bn
conv2_3_3x3_relu = mx.symbol.Activation(name='conv2_3_3x3_relu', data=conv2_3_3x3_bn_scale , act_type='relu')
conv2_3_1x1_increase = mx.symbol.Convolution(name='conv2_3_1x1_increase', data=conv2_3_3x3_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv2_3_1x1_increase_bn = mx.symbol.BatchNorm(name='conv2_3_1x1_increase_bn', data=conv2_3_1x1_increase , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv2_3_1x1_increase_bn_scale = conv2_3_1x1_increase_bn
conv2_3_global_pool = mx.symbol.Pooling(name='conv2_3_global_pool', data=conv2_3_1x1_increase_bn_scale , pooling_convention='full', global_pool=True, kernel=(1,1), pool_type='avg')
conv2_3_1x1_down = mx.symbol.Convolution(name='conv2_3_1x1_down', data=conv2_3_global_pool , num_filter=16, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv2_3_1x1_down_relu = mx.symbol.Activation(name='conv2_3_1x1_down_relu', data=conv2_3_1x1_down , act_type='relu')
conv2_3_1x1_up = mx.symbol.Convolution(name='conv2_3_1x1_up', data=conv2_3_1x1_down_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv2_3_prob = mx.symbol.Activation(name='conv2_3_prob', data=conv2_3_1x1_up , act_type='sigmoid')
if memonger:
    conv2_2_relu._set_attr(mirror_stage='True')
conv2_3 = mx.sym.broadcast_mul(conv2_3_prob, conv2_3_1x1_increase_bn_scale) + conv2_2_relu
conv2_3_relu = mx.symbol.Activation(name='conv2_3_relu', data=conv2_3 , act_type='relu')
conv3_1_1x1_reduce = mx.symbol.Convolution(name='conv3_1_1x1_reduce', data=conv2_3_relu , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
conv3_1_1x1_reduce_bn = mx.symbol.BatchNorm(name='conv3_1_1x1_reduce_bn', data=conv3_1_1x1_reduce , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv3_1_1x1_reduce_bn_scale = conv3_1_1x1_reduce_bn
conv3_1_1x1_reduce_relu = mx.symbol.Activation(name='conv3_1_1x1_reduce_relu', data=conv3_1_1x1_reduce_bn_scale , act_type='relu')
conv3_1_3x3 = mx.symbol.Convolution(name='conv3_1_3x3', data=conv3_1_1x1_reduce_relu , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
conv3_1_3x3_bn = mx.symbol.BatchNorm(name='conv3_1_3x3_bn', data=conv3_1_3x3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv3_1_3x3_bn_scale = conv3_1_3x3_bn
conv3_1_3x3_relu = mx.symbol.Activation(name='conv3_1_3x3_relu', data=conv3_1_3x3_bn_scale , act_type='relu')
conv3_1_1x1_increase = mx.symbol.Convolution(name='conv3_1_1x1_increase', data=conv3_1_3x3_relu , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv3_1_1x1_increase_bn = mx.symbol.BatchNorm(name='conv3_1_1x1_increase_bn', data=conv3_1_1x1_increase , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv3_1_1x1_increase_bn_scale = conv3_1_1x1_increase_bn
conv3_1_global_pool = mx.symbol.Pooling(name='conv3_1_global_pool', data=conv3_1_1x1_increase_bn_scale , pooling_convention='full', global_pool=True, kernel=(1,1), pool_type='avg')
conv3_1_1x1_down = mx.symbol.Convolution(name='conv3_1_1x1_down', data=conv3_1_global_pool , num_filter=32, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv3_1_1x1_down_relu = mx.symbol.Activation(name='conv3_1_1x1_down_relu', data=conv3_1_1x1_down , act_type='relu')
conv3_1_1x1_up = mx.symbol.Convolution(name='conv3_1_1x1_up', data=conv3_1_1x1_down_relu , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv3_1_prob = mx.symbol.Activation(name='conv3_1_prob', data=conv3_1_1x1_up , act_type='sigmoid')
conv3_1_1x1_proj = mx.symbol.Convolution(name='conv3_1_1x1_proj', data=conv2_3_relu , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
conv3_1_1x1_proj_bn = mx.symbol.BatchNorm(name='conv3_1_1x1_proj_bn', data=conv3_1_1x1_proj , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv3_1_1x1_proj_bn_scale = conv3_1_1x1_proj_bn
if memonger:
    conv3_1_1x1_proj_bn_scale._set_attr(mirror_stage='True')
conv3_1 = mx.sym.broadcast_mul(conv3_1_prob, conv3_1_1x1_increase_bn_scale) + conv3_1_1x1_proj_bn_scale
conv3_1_relu = mx.symbol.Activation(name='conv3_1_relu', data=conv3_1 , act_type='relu')
conv3_2_1x1_reduce = mx.symbol.Convolution(name='conv3_2_1x1_reduce', data=conv3_1_relu , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv3_2_1x1_reduce_bn = mx.symbol.BatchNorm(name='conv3_2_1x1_reduce_bn', data=conv3_2_1x1_reduce , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv3_2_1x1_reduce_bn_scale = conv3_2_1x1_reduce_bn
conv3_2_1x1_reduce_relu = mx.symbol.Activation(name='conv3_2_1x1_reduce_relu', data=conv3_2_1x1_reduce_bn_scale , act_type='relu')
conv3_2_3x3 = mx.symbol.Convolution(name='conv3_2_3x3', data=conv3_2_1x1_reduce_relu , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
conv3_2_3x3_bn = mx.symbol.BatchNorm(name='conv3_2_3x3_bn', data=conv3_2_3x3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv3_2_3x3_bn_scale = conv3_2_3x3_bn
conv3_2_3x3_relu = mx.symbol.Activation(name='conv3_2_3x3_relu', data=conv3_2_3x3_bn_scale , act_type='relu')
conv3_2_1x1_increase = mx.symbol.Convolution(name='conv3_2_1x1_increase', data=conv3_2_3x3_relu , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv3_2_1x1_increase_bn = mx.symbol.BatchNorm(name='conv3_2_1x1_increase_bn', data=conv3_2_1x1_increase , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv3_2_1x1_increase_bn_scale = conv3_2_1x1_increase_bn
conv3_2_global_pool = mx.symbol.Pooling(name='conv3_2_global_pool', data=conv3_2_1x1_increase_bn_scale , pooling_convention='full', global_pool=True, kernel=(1,1), pool_type='avg')
conv3_2_1x1_down = mx.symbol.Convolution(name='conv3_2_1x1_down', data=conv3_2_global_pool , num_filter=32, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv3_2_1x1_down_relu = mx.symbol.Activation(name='conv3_2_1x1_down_relu', data=conv3_2_1x1_down , act_type='relu')
conv3_2_1x1_up = mx.symbol.Convolution(name='conv3_2_1x1_up', data=conv3_2_1x1_down_relu , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv3_2_prob = mx.symbol.Activation(name='conv3_2_prob', data=conv3_2_1x1_up , act_type='sigmoid')
if memonger:
    conv3_1_relu._set_attr(mirror_stage='True')
conv3_2 = mx.sym.broadcast_mul(conv3_2_prob, conv3_2_1x1_increase_bn_scale) + conv3_1_relu
conv3_2_relu = mx.symbol.Activation(name='conv3_2_relu', data=conv3_2 , act_type='relu')
conv3_3_1x1_reduce = mx.symbol.Convolution(name='conv3_3_1x1_reduce', data=conv3_2_relu , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv3_3_1x1_reduce_bn = mx.symbol.BatchNorm(name='conv3_3_1x1_reduce_bn', data=conv3_3_1x1_reduce , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv3_3_1x1_reduce_bn_scale = conv3_3_1x1_reduce_bn
conv3_3_1x1_reduce_relu = mx.symbol.Activation(name='conv3_3_1x1_reduce_relu', data=conv3_3_1x1_reduce_bn_scale , act_type='relu')
conv3_3_3x3 = mx.symbol.Convolution(name='conv3_3_3x3', data=conv3_3_1x1_reduce_relu , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
conv3_3_3x3_bn = mx.symbol.BatchNorm(name='conv3_3_3x3_bn', data=conv3_3_3x3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv3_3_3x3_bn_scale = conv3_3_3x3_bn
conv3_3_3x3_relu = mx.symbol.Activation(name='conv3_3_3x3_relu', data=conv3_3_3x3_bn_scale , act_type='relu')
conv3_3_1x1_increase = mx.symbol.Convolution(name='conv3_3_1x1_increase', data=conv3_3_3x3_relu , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv3_3_1x1_increase_bn = mx.symbol.BatchNorm(name='conv3_3_1x1_increase_bn', data=conv3_3_1x1_increase , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv3_3_1x1_increase_bn_scale = conv3_3_1x1_increase_bn
conv3_3_global_pool = mx.symbol.Pooling(name='conv3_3_global_pool', data=conv3_3_1x1_increase_bn_scale , pooling_convention='full', global_pool=True, kernel=(1,1), pool_type='avg')
conv3_3_1x1_down = mx.symbol.Convolution(name='conv3_3_1x1_down', data=conv3_3_global_pool , num_filter=32, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv3_3_1x1_down_relu = mx.symbol.Activation(name='conv3_3_1x1_down_relu', data=conv3_3_1x1_down , act_type='relu')
conv3_3_1x1_up = mx.symbol.Convolution(name='conv3_3_1x1_up', data=conv3_3_1x1_down_relu , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv3_3_prob = mx.symbol.Activation(name='conv3_3_prob', data=conv3_3_1x1_up , act_type='sigmoid')
if memonger:
    conv3_2_relu._set_attr(mirror_stage='True')
conv3_3 = mx.sym.broadcast_mul(conv3_3_prob, conv3_3_1x1_increase_bn_scale) + conv3_2_relu
conv3_3_relu = mx.symbol.Activation(name='conv3_3_relu', data=conv3_3 , act_type='relu')
conv3_4_1x1_reduce = mx.symbol.Convolution(name='conv3_4_1x1_reduce', data=conv3_3_relu , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv3_4_1x1_reduce_bn = mx.symbol.BatchNorm(name='conv3_4_1x1_reduce_bn', data=conv3_4_1x1_reduce , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv3_4_1x1_reduce_bn_scale = conv3_4_1x1_reduce_bn
conv3_4_1x1_reduce_relu = mx.symbol.Activation(name='conv3_4_1x1_reduce_relu', data=conv3_4_1x1_reduce_bn_scale , act_type='relu')
conv3_4_3x3 = mx.symbol.Convolution(name='conv3_4_3x3', data=conv3_4_1x1_reduce_relu , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
conv3_4_3x3_bn = mx.symbol.BatchNorm(name='conv3_4_3x3_bn', data=conv3_4_3x3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv3_4_3x3_bn_scale = conv3_4_3x3_bn
conv3_4_3x3_relu = mx.symbol.Activation(name='conv3_4_3x3_relu', data=conv3_4_3x3_bn_scale , act_type='relu')
conv3_4_1x1_increase = mx.symbol.Convolution(name='conv3_4_1x1_increase', data=conv3_4_3x3_relu , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv3_4_1x1_increase_bn = mx.symbol.BatchNorm(name='conv3_4_1x1_increase_bn', data=conv3_4_1x1_increase , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv3_4_1x1_increase_bn_scale = conv3_4_1x1_increase_bn
conv3_4_global_pool = mx.symbol.Pooling(name='conv3_4_global_pool', data=conv3_4_1x1_increase_bn_scale , pooling_convention='full', global_pool=True, kernel=(1,1), pool_type='avg')
conv3_4_1x1_down = mx.symbol.Convolution(name='conv3_4_1x1_down', data=conv3_4_global_pool , num_filter=32, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv3_4_1x1_down_relu = mx.symbol.Activation(name='conv3_4_1x1_down_relu', data=conv3_4_1x1_down , act_type='relu')
conv3_4_1x1_up = mx.symbol.Convolution(name='conv3_4_1x1_up', data=conv3_4_1x1_down_relu , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv3_4_prob = mx.symbol.Activation(name='conv3_4_prob', data=conv3_4_1x1_up , act_type='sigmoid')
if memonger:
    conv3_3_relu._set_attr(mirror_stage='True')
conv3_4 = mx.sym.broadcast_mul(conv3_4_prob, conv3_4_1x1_increase_bn_scale) + conv3_3_relu
conv3_4_relu = mx.symbol.Activation(name='conv3_4_relu', data=conv3_4 , act_type='relu')
conv4_1_1x1_reduce = mx.symbol.Convolution(name='conv4_1_1x1_reduce', data=conv3_4_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
conv4_1_1x1_reduce_bn = mx.symbol.BatchNorm(name='conv4_1_1x1_reduce_bn', data=conv4_1_1x1_reduce , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_1_1x1_reduce_bn_scale = conv4_1_1x1_reduce_bn
conv4_1_1x1_reduce_relu = mx.symbol.Activation(name='conv4_1_1x1_reduce_relu', data=conv4_1_1x1_reduce_bn_scale , act_type='relu')
conv4_1_3x3 = mx.symbol.Convolution(name='conv4_1_3x3', data=conv4_1_1x1_reduce_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
conv4_1_3x3_bn = mx.symbol.BatchNorm(name='conv4_1_3x3_bn', data=conv4_1_3x3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_1_3x3_bn_scale = conv4_1_3x3_bn
conv4_1_3x3_relu = mx.symbol.Activation(name='conv4_1_3x3_relu', data=conv4_1_3x3_bn_scale , act_type='relu')
conv4_1_1x1_increase = mx.symbol.Convolution(name='conv4_1_1x1_increase', data=conv4_1_3x3_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv4_1_1x1_increase_bn = mx.symbol.BatchNorm(name='conv4_1_1x1_increase_bn', data=conv4_1_1x1_increase , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_1_1x1_increase_bn_scale = conv4_1_1x1_increase_bn
conv4_1_global_pool = mx.symbol.Pooling(name='conv4_1_global_pool', data=conv4_1_1x1_increase_bn_scale , pooling_convention='full', global_pool=True, kernel=(1,1), pool_type='avg')
conv4_1_1x1_down = mx.symbol.Convolution(name='conv4_1_1x1_down', data=conv4_1_global_pool , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv4_1_1x1_down_relu = mx.symbol.Activation(name='conv4_1_1x1_down_relu', data=conv4_1_1x1_down , act_type='relu')
conv4_1_1x1_up = mx.symbol.Convolution(name='conv4_1_1x1_up', data=conv4_1_1x1_down_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv4_1_prob = mx.symbol.Activation(name='conv4_1_prob', data=conv4_1_1x1_up , act_type='sigmoid')
conv4_1_1x1_proj = mx.symbol.Convolution(name='conv4_1_1x1_proj', data=conv3_4_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
conv4_1_1x1_proj_bn = mx.symbol.BatchNorm(name='conv4_1_1x1_proj_bn', data=conv4_1_1x1_proj , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_1_1x1_proj_bn_scale = conv4_1_1x1_proj_bn
if memonger:
    conv4_1_1x1_proj_bn_scale._set_attr(mirror_stage='True')
conv4_1 = mx.sym.broadcast_mul(conv4_1_prob, conv4_1_1x1_increase_bn_scale) + conv4_1_1x1_proj_bn_scale
conv4_1_relu = mx.symbol.Activation(name='conv4_1_relu', data=conv4_1 , act_type='relu')
conv4_2_1x1_reduce = mx.symbol.Convolution(name='conv4_2_1x1_reduce', data=conv4_1_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv4_2_1x1_reduce_bn = mx.symbol.BatchNorm(name='conv4_2_1x1_reduce_bn', data=conv4_2_1x1_reduce , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_2_1x1_reduce_bn_scale = conv4_2_1x1_reduce_bn
conv4_2_1x1_reduce_relu = mx.symbol.Activation(name='conv4_2_1x1_reduce_relu', data=conv4_2_1x1_reduce_bn_scale , act_type='relu')
conv4_2_3x3 = mx.symbol.Convolution(name='conv4_2_3x3', data=conv4_2_1x1_reduce_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
conv4_2_3x3_bn = mx.symbol.BatchNorm(name='conv4_2_3x3_bn', data=conv4_2_3x3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_2_3x3_bn_scale = conv4_2_3x3_bn
conv4_2_3x3_relu = mx.symbol.Activation(name='conv4_2_3x3_relu', data=conv4_2_3x3_bn_scale , act_type='relu')
conv4_2_1x1_increase = mx.symbol.Convolution(name='conv4_2_1x1_increase', data=conv4_2_3x3_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv4_2_1x1_increase_bn = mx.symbol.BatchNorm(name='conv4_2_1x1_increase_bn', data=conv4_2_1x1_increase , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_2_1x1_increase_bn_scale = conv4_2_1x1_increase_bn
conv4_2_global_pool = mx.symbol.Pooling(name='conv4_2_global_pool', data=conv4_2_1x1_increase_bn_scale , pooling_convention='full', global_pool=True, kernel=(1,1), pool_type='avg')
conv4_2_1x1_down = mx.symbol.Convolution(name='conv4_2_1x1_down', data=conv4_2_global_pool , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv4_2_1x1_down_relu = mx.symbol.Activation(name='conv4_2_1x1_down_relu', data=conv4_2_1x1_down , act_type='relu')
conv4_2_1x1_up = mx.symbol.Convolution(name='conv4_2_1x1_up', data=conv4_2_1x1_down_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv4_2_prob = mx.symbol.Activation(name='conv4_2_prob', data=conv4_2_1x1_up , act_type='sigmoid')
if memonger:
    conv4_1_relu._set_attr(mirror_stage='True')
conv4_2 = mx.sym.broadcast_mul(conv4_2_prob, conv4_2_1x1_increase_bn_scale) + conv4_1_relu
conv4_2_relu = mx.symbol.Activation(name='conv4_2_relu', data=conv4_2 , act_type='relu')
conv4_3_1x1_reduce = mx.symbol.Convolution(name='conv4_3_1x1_reduce', data=conv4_2_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv4_3_1x1_reduce_bn = mx.symbol.BatchNorm(name='conv4_3_1x1_reduce_bn', data=conv4_3_1x1_reduce , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_3_1x1_reduce_bn_scale = conv4_3_1x1_reduce_bn
conv4_3_1x1_reduce_relu = mx.symbol.Activation(name='conv4_3_1x1_reduce_relu', data=conv4_3_1x1_reduce_bn_scale , act_type='relu')
conv4_3_3x3 = mx.symbol.Convolution(name='conv4_3_3x3', data=conv4_3_1x1_reduce_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
conv4_3_3x3_bn = mx.symbol.BatchNorm(name='conv4_3_3x3_bn', data=conv4_3_3x3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_3_3x3_bn_scale = conv4_3_3x3_bn
conv4_3_3x3_relu = mx.symbol.Activation(name='conv4_3_3x3_relu', data=conv4_3_3x3_bn_scale , act_type='relu')
conv4_3_1x1_increase = mx.symbol.Convolution(name='conv4_3_1x1_increase', data=conv4_3_3x3_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv4_3_1x1_increase_bn = mx.symbol.BatchNorm(name='conv4_3_1x1_increase_bn', data=conv4_3_1x1_increase , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_3_1x1_increase_bn_scale = conv4_3_1x1_increase_bn
conv4_3_global_pool = mx.symbol.Pooling(name='conv4_3_global_pool', data=conv4_3_1x1_increase_bn_scale , pooling_convention='full', global_pool=True, kernel=(1,1), pool_type='avg')
conv4_3_1x1_down = mx.symbol.Convolution(name='conv4_3_1x1_down', data=conv4_3_global_pool , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv4_3_1x1_down_relu = mx.symbol.Activation(name='conv4_3_1x1_down_relu', data=conv4_3_1x1_down , act_type='relu')
conv4_3_1x1_up = mx.symbol.Convolution(name='conv4_3_1x1_up', data=conv4_3_1x1_down_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv4_3_prob = mx.symbol.Activation(name='conv4_3_prob', data=conv4_3_1x1_up , act_type='sigmoid')
if memonger:
    conv4_2_relu._set_attr(mirror_stage='True')
conv4_3 = mx.sym.broadcast_mul(conv4_3_prob, conv4_3_1x1_increase_bn_scale) + conv4_2_relu
conv4_3_relu = mx.symbol.Activation(name='conv4_3_relu', data=conv4_3 , act_type='relu')
conv4_4_1x1_reduce = mx.symbol.Convolution(name='conv4_4_1x1_reduce', data=conv4_3_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv4_4_1x1_reduce_bn = mx.symbol.BatchNorm(name='conv4_4_1x1_reduce_bn', data=conv4_4_1x1_reduce , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_4_1x1_reduce_bn_scale = conv4_4_1x1_reduce_bn
conv4_4_1x1_reduce_relu = mx.symbol.Activation(name='conv4_4_1x1_reduce_relu', data=conv4_4_1x1_reduce_bn_scale , act_type='relu')
conv4_4_3x3 = mx.symbol.Convolution(name='conv4_4_3x3', data=conv4_4_1x1_reduce_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
conv4_4_3x3_bn = mx.symbol.BatchNorm(name='conv4_4_3x3_bn', data=conv4_4_3x3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_4_3x3_bn_scale = conv4_4_3x3_bn
conv4_4_3x3_relu = mx.symbol.Activation(name='conv4_4_3x3_relu', data=conv4_4_3x3_bn_scale , act_type='relu')
conv4_4_1x1_increase = mx.symbol.Convolution(name='conv4_4_1x1_increase', data=conv4_4_3x3_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv4_4_1x1_increase_bn = mx.symbol.BatchNorm(name='conv4_4_1x1_increase_bn', data=conv4_4_1x1_increase , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_4_1x1_increase_bn_scale = conv4_4_1x1_increase_bn
conv4_4_global_pool = mx.symbol.Pooling(name='conv4_4_global_pool', data=conv4_4_1x1_increase_bn_scale , pooling_convention='full', global_pool=True, kernel=(1,1), pool_type='avg')
conv4_4_1x1_down = mx.symbol.Convolution(name='conv4_4_1x1_down', data=conv4_4_global_pool , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv4_4_1x1_down_relu = mx.symbol.Activation(name='conv4_4_1x1_down_relu', data=conv4_4_1x1_down , act_type='relu')
conv4_4_1x1_up = mx.symbol.Convolution(name='conv4_4_1x1_up', data=conv4_4_1x1_down_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv4_4_prob = mx.symbol.Activation(name='conv4_4_prob', data=conv4_4_1x1_up , act_type='sigmoid')
if memonger:
    conv4_3_relu._set_attr(mirror_stage='True')
conv4_4 = mx.sym.broadcast_mul(conv4_4_prob, conv4_4_1x1_increase_bn_scale) + conv4_3_relu
conv4_4_relu = mx.symbol.Activation(name='conv4_4_relu', data=conv4_4 , act_type='relu')
conv4_5_1x1_reduce = mx.symbol.Convolution(name='conv4_5_1x1_reduce', data=conv4_4_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv4_5_1x1_reduce_bn = mx.symbol.BatchNorm(name='conv4_5_1x1_reduce_bn', data=conv4_5_1x1_reduce , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_5_1x1_reduce_bn_scale = conv4_5_1x1_reduce_bn
conv4_5_1x1_reduce_relu = mx.symbol.Activation(name='conv4_5_1x1_reduce_relu', data=conv4_5_1x1_reduce_bn_scale , act_type='relu')
conv4_5_3x3 = mx.symbol.Convolution(name='conv4_5_3x3', data=conv4_5_1x1_reduce_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
conv4_5_3x3_bn = mx.symbol.BatchNorm(name='conv4_5_3x3_bn', data=conv4_5_3x3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_5_3x3_bn_scale = conv4_5_3x3_bn
conv4_5_3x3_relu = mx.symbol.Activation(name='conv4_5_3x3_relu', data=conv4_5_3x3_bn_scale , act_type='relu')
conv4_5_1x1_increase = mx.symbol.Convolution(name='conv4_5_1x1_increase', data=conv4_5_3x3_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv4_5_1x1_increase_bn = mx.symbol.BatchNorm(name='conv4_5_1x1_increase_bn', data=conv4_5_1x1_increase , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_5_1x1_increase_bn_scale = conv4_5_1x1_increase_bn
conv4_5_global_pool = mx.symbol.Pooling(name='conv4_5_global_pool', data=conv4_5_1x1_increase_bn_scale , pooling_convention='full', global_pool=True, kernel=(1,1), pool_type='avg')
conv4_5_1x1_down = mx.symbol.Convolution(name='conv4_5_1x1_down', data=conv4_5_global_pool , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv4_5_1x1_down_relu = mx.symbol.Activation(name='conv4_5_1x1_down_relu', data=conv4_5_1x1_down , act_type='relu')
conv4_5_1x1_up = mx.symbol.Convolution(name='conv4_5_1x1_up', data=conv4_5_1x1_down_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv4_5_prob = mx.symbol.Activation(name='conv4_5_prob', data=conv4_5_1x1_up , act_type='sigmoid')
if memonger:
    conv4_4_relu._set_attr(mirror_stage='True')
conv4_5 = mx.sym.broadcast_mul(conv4_5_prob, conv4_5_1x1_increase_bn_scale) + conv4_4_relu
conv4_5_relu = mx.symbol.Activation(name='conv4_5_relu', data=conv4_5 , act_type='relu')
conv4_6_1x1_reduce = mx.symbol.Convolution(name='conv4_6_1x1_reduce', data=conv4_5_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv4_6_1x1_reduce_bn = mx.symbol.BatchNorm(name='conv4_6_1x1_reduce_bn', data=conv4_6_1x1_reduce , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_6_1x1_reduce_bn_scale = conv4_6_1x1_reduce_bn
conv4_6_1x1_reduce_relu = mx.symbol.Activation(name='conv4_6_1x1_reduce_relu', data=conv4_6_1x1_reduce_bn_scale , act_type='relu')
conv4_6_3x3 = mx.symbol.Convolution(name='conv4_6_3x3', data=conv4_6_1x1_reduce_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
conv4_6_3x3_bn = mx.symbol.BatchNorm(name='conv4_6_3x3_bn', data=conv4_6_3x3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_6_3x3_bn_scale = conv4_6_3x3_bn
conv4_6_3x3_relu = mx.symbol.Activation(name='conv4_6_3x3_relu', data=conv4_6_3x3_bn_scale , act_type='relu')
conv4_6_1x1_increase = mx.symbol.Convolution(name='conv4_6_1x1_increase', data=conv4_6_3x3_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv4_6_1x1_increase_bn = mx.symbol.BatchNorm(name='conv4_6_1x1_increase_bn', data=conv4_6_1x1_increase , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_6_1x1_increase_bn_scale = conv4_6_1x1_increase_bn
conv4_6_global_pool = mx.symbol.Pooling(name='conv4_6_global_pool', data=conv4_6_1x1_increase_bn_scale , pooling_convention='full', global_pool=True, kernel=(1,1), pool_type='avg')
conv4_6_1x1_down = mx.symbol.Convolution(name='conv4_6_1x1_down', data=conv4_6_global_pool , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv4_6_1x1_down_relu = mx.symbol.Activation(name='conv4_6_1x1_down_relu', data=conv4_6_1x1_down , act_type='relu')
conv4_6_1x1_up = mx.symbol.Convolution(name='conv4_6_1x1_up', data=conv4_6_1x1_down_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv4_6_prob = mx.symbol.Activation(name='conv4_6_prob', data=conv4_6_1x1_up , act_type='sigmoid')
if memonger:
    conv4_5_relu._set_attr(mirror_stage='True')
conv4_6 = mx.sym.broadcast_mul(conv4_6_prob, conv4_6_1x1_increase_bn_scale) + conv4_5_relu
conv4_6_relu = mx.symbol.Activation(name='conv4_6_relu', data=conv4_6 , act_type='relu')
conv4_7_1x1_reduce = mx.symbol.Convolution(name='conv4_7_1x1_reduce', data=conv4_6_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv4_7_1x1_reduce_bn = mx.symbol.BatchNorm(name='conv4_7_1x1_reduce_bn', data=conv4_7_1x1_reduce , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_7_1x1_reduce_bn_scale = conv4_7_1x1_reduce_bn
conv4_7_1x1_reduce_relu = mx.symbol.Activation(name='conv4_7_1x1_reduce_relu', data=conv4_7_1x1_reduce_bn_scale , act_type='relu')
conv4_7_3x3 = mx.symbol.Convolution(name='conv4_7_3x3', data=conv4_7_1x1_reduce_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
conv4_7_3x3_bn = mx.symbol.BatchNorm(name='conv4_7_3x3_bn', data=conv4_7_3x3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_7_3x3_bn_scale = conv4_7_3x3_bn
conv4_7_3x3_relu = mx.symbol.Activation(name='conv4_7_3x3_relu', data=conv4_7_3x3_bn_scale , act_type='relu')
conv4_7_1x1_increase = mx.symbol.Convolution(name='conv4_7_1x1_increase', data=conv4_7_3x3_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv4_7_1x1_increase_bn = mx.symbol.BatchNorm(name='conv4_7_1x1_increase_bn', data=conv4_7_1x1_increase , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_7_1x1_increase_bn_scale = conv4_7_1x1_increase_bn
conv4_7_global_pool = mx.symbol.Pooling(name='conv4_7_global_pool', data=conv4_7_1x1_increase_bn_scale , pooling_convention='full', global_pool=True, kernel=(1,1), pool_type='avg')
conv4_7_1x1_down = mx.symbol.Convolution(name='conv4_7_1x1_down', data=conv4_7_global_pool , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv4_7_1x1_down_relu = mx.symbol.Activation(name='conv4_7_1x1_down_relu', data=conv4_7_1x1_down , act_type='relu')
conv4_7_1x1_up = mx.symbol.Convolution(name='conv4_7_1x1_up', data=conv4_7_1x1_down_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv4_7_prob = mx.symbol.Activation(name='conv4_7_prob', data=conv4_7_1x1_up , act_type='sigmoid')
if memonger:
    conv4_6_relu._set_attr(mirror_stage='True')
conv4_7 = mx.sym.broadcast_mul(conv4_7_prob, conv4_7_1x1_increase_bn_scale) + conv4_6_relu
conv4_7_relu = mx.symbol.Activation(name='conv4_7_relu', data=conv4_7 , act_type='relu')
conv4_8_1x1_reduce = mx.symbol.Convolution(name='conv4_8_1x1_reduce', data=conv4_7_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv4_8_1x1_reduce_bn = mx.symbol.BatchNorm(name='conv4_8_1x1_reduce_bn', data=conv4_8_1x1_reduce , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_8_1x1_reduce_bn_scale = conv4_8_1x1_reduce_bn
conv4_8_1x1_reduce_relu = mx.symbol.Activation(name='conv4_8_1x1_reduce_relu', data=conv4_8_1x1_reduce_bn_scale , act_type='relu')
conv4_8_3x3 = mx.symbol.Convolution(name='conv4_8_3x3', data=conv4_8_1x1_reduce_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
conv4_8_3x3_bn = mx.symbol.BatchNorm(name='conv4_8_3x3_bn', data=conv4_8_3x3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_8_3x3_bn_scale = conv4_8_3x3_bn
conv4_8_3x3_relu = mx.symbol.Activation(name='conv4_8_3x3_relu', data=conv4_8_3x3_bn_scale , act_type='relu')
conv4_8_1x1_increase = mx.symbol.Convolution(name='conv4_8_1x1_increase', data=conv4_8_3x3_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv4_8_1x1_increase_bn = mx.symbol.BatchNorm(name='conv4_8_1x1_increase_bn', data=conv4_8_1x1_increase , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_8_1x1_increase_bn_scale = conv4_8_1x1_increase_bn
conv4_8_global_pool = mx.symbol.Pooling(name='conv4_8_global_pool', data=conv4_8_1x1_increase_bn_scale , pooling_convention='full', global_pool=True, kernel=(1,1), pool_type='avg')
conv4_8_1x1_down = mx.symbol.Convolution(name='conv4_8_1x1_down', data=conv4_8_global_pool , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv4_8_1x1_down_relu = mx.symbol.Activation(name='conv4_8_1x1_down_relu', data=conv4_8_1x1_down , act_type='relu')
conv4_8_1x1_up = mx.symbol.Convolution(name='conv4_8_1x1_up', data=conv4_8_1x1_down_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv4_8_prob = mx.symbol.Activation(name='conv4_8_prob', data=conv4_8_1x1_up , act_type='sigmoid')
if memonger:
    conv4_7_relu._set_attr(mirror_stage='True')
conv4_8 = mx.sym.broadcast_mul(conv4_8_prob, conv4_8_1x1_increase_bn_scale) + conv4_7_relu
conv4_8_relu = mx.symbol.Activation(name='conv4_8_relu', data=conv4_8 , act_type='relu')
conv4_9_1x1_reduce = mx.symbol.Convolution(name='conv4_9_1x1_reduce', data=conv4_8_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv4_9_1x1_reduce_bn = mx.symbol.BatchNorm(name='conv4_9_1x1_reduce_bn', data=conv4_9_1x1_reduce , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_9_1x1_reduce_bn_scale = conv4_9_1x1_reduce_bn
conv4_9_1x1_reduce_relu = mx.symbol.Activation(name='conv4_9_1x1_reduce_relu', data=conv4_9_1x1_reduce_bn_scale , act_type='relu')
conv4_9_3x3 = mx.symbol.Convolution(name='conv4_9_3x3', data=conv4_9_1x1_reduce_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
conv4_9_3x3_bn = mx.symbol.BatchNorm(name='conv4_9_3x3_bn', data=conv4_9_3x3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_9_3x3_bn_scale = conv4_9_3x3_bn
conv4_9_3x3_relu = mx.symbol.Activation(name='conv4_9_3x3_relu', data=conv4_9_3x3_bn_scale , act_type='relu')
conv4_9_1x1_increase = mx.symbol.Convolution(name='conv4_9_1x1_increase', data=conv4_9_3x3_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv4_9_1x1_increase_bn = mx.symbol.BatchNorm(name='conv4_9_1x1_increase_bn', data=conv4_9_1x1_increase , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_9_1x1_increase_bn_scale = conv4_9_1x1_increase_bn
conv4_9_global_pool = mx.symbol.Pooling(name='conv4_9_global_pool', data=conv4_9_1x1_increase_bn_scale , pooling_convention='full', global_pool=True, kernel=(1,1), pool_type='avg')
conv4_9_1x1_down = mx.symbol.Convolution(name='conv4_9_1x1_down', data=conv4_9_global_pool , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv4_9_1x1_down_relu = mx.symbol.Activation(name='conv4_9_1x1_down_relu', data=conv4_9_1x1_down , act_type='relu')
conv4_9_1x1_up = mx.symbol.Convolution(name='conv4_9_1x1_up', data=conv4_9_1x1_down_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv4_9_prob = mx.symbol.Activation(name='conv4_9_prob', data=conv4_9_1x1_up , act_type='sigmoid')
if memonger:
    conv4_8_relu._set_attr(mirror_stage='True')
conv4_9 = mx.sym.broadcast_mul(conv4_9_prob, conv4_9_1x1_increase_bn_scale) + conv4_8_relu
conv4_9_relu = mx.symbol.Activation(name='conv4_9_relu', data=conv4_9 , act_type='relu')
conv4_10_1x1_reduce = mx.symbol.Convolution(name='conv4_10_1x1_reduce', data=conv4_9_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv4_10_1x1_reduce_bn = mx.symbol.BatchNorm(name='conv4_10_1x1_reduce_bn', data=conv4_10_1x1_reduce , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_10_1x1_reduce_bn_scale = conv4_10_1x1_reduce_bn
conv4_10_1x1_reduce_relu = mx.symbol.Activation(name='conv4_10_1x1_reduce_relu', data=conv4_10_1x1_reduce_bn_scale , act_type='relu')
conv4_10_3x3 = mx.symbol.Convolution(name='conv4_10_3x3', data=conv4_10_1x1_reduce_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
conv4_10_3x3_bn = mx.symbol.BatchNorm(name='conv4_10_3x3_bn', data=conv4_10_3x3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_10_3x3_bn_scale = conv4_10_3x3_bn
conv4_10_3x3_relu = mx.symbol.Activation(name='conv4_10_3x3_relu', data=conv4_10_3x3_bn_scale , act_type='relu')
conv4_10_1x1_increase = mx.symbol.Convolution(name='conv4_10_1x1_increase', data=conv4_10_3x3_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv4_10_1x1_increase_bn = mx.symbol.BatchNorm(name='conv4_10_1x1_increase_bn', data=conv4_10_1x1_increase , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_10_1x1_increase_bn_scale = conv4_10_1x1_increase_bn
conv4_10_global_pool = mx.symbol.Pooling(name='conv4_10_global_pool', data=conv4_10_1x1_increase_bn_scale , pooling_convention='full', global_pool=True, kernel=(1,1), pool_type='avg')
conv4_10_1x1_down = mx.symbol.Convolution(name='conv4_10_1x1_down', data=conv4_10_global_pool , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv4_10_1x1_down_relu = mx.symbol.Activation(name='conv4_10_1x1_down_relu', data=conv4_10_1x1_down , act_type='relu')
conv4_10_1x1_up = mx.symbol.Convolution(name='conv4_10_1x1_up', data=conv4_10_1x1_down_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv4_10_prob = mx.symbol.Activation(name='conv4_10_prob', data=conv4_10_1x1_up , act_type='sigmoid')
if memonger:
    conv4_9_relu._set_attr(mirror_stage='True')
conv4_10 = mx.sym.broadcast_mul(conv4_10_prob, conv4_10_1x1_increase_bn_scale) + conv4_9_relu
conv4_10_relu = mx.symbol.Activation(name='conv4_10_relu', data=conv4_10 , act_type='relu')
conv4_11_1x1_reduce = mx.symbol.Convolution(name='conv4_11_1x1_reduce', data=conv4_10_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv4_11_1x1_reduce_bn = mx.symbol.BatchNorm(name='conv4_11_1x1_reduce_bn', data=conv4_11_1x1_reduce , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_11_1x1_reduce_bn_scale = conv4_11_1x1_reduce_bn
conv4_11_1x1_reduce_relu = mx.symbol.Activation(name='conv4_11_1x1_reduce_relu', data=conv4_11_1x1_reduce_bn_scale , act_type='relu')
conv4_11_3x3 = mx.symbol.Convolution(name='conv4_11_3x3', data=conv4_11_1x1_reduce_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
conv4_11_3x3_bn = mx.symbol.BatchNorm(name='conv4_11_3x3_bn', data=conv4_11_3x3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_11_3x3_bn_scale = conv4_11_3x3_bn
conv4_11_3x3_relu = mx.symbol.Activation(name='conv4_11_3x3_relu', data=conv4_11_3x3_bn_scale , act_type='relu')
conv4_11_1x1_increase = mx.symbol.Convolution(name='conv4_11_1x1_increase', data=conv4_11_3x3_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv4_11_1x1_increase_bn = mx.symbol.BatchNorm(name='conv4_11_1x1_increase_bn', data=conv4_11_1x1_increase , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_11_1x1_increase_bn_scale = conv4_11_1x1_increase_bn
conv4_11_global_pool = mx.symbol.Pooling(name='conv4_11_global_pool', data=conv4_11_1x1_increase_bn_scale , pooling_convention='full', global_pool=True, kernel=(1,1), pool_type='avg')
conv4_11_1x1_down = mx.symbol.Convolution(name='conv4_11_1x1_down', data=conv4_11_global_pool , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv4_11_1x1_down_relu = mx.symbol.Activation(name='conv4_11_1x1_down_relu', data=conv4_11_1x1_down , act_type='relu')
conv4_11_1x1_up = mx.symbol.Convolution(name='conv4_11_1x1_up', data=conv4_11_1x1_down_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv4_11_prob = mx.symbol.Activation(name='conv4_11_prob', data=conv4_11_1x1_up , act_type='sigmoid')
if memonger:
    conv4_10_relu._set_attr(mirror_stage='True')
conv4_11 = mx.sym.broadcast_mul(conv4_11_prob, conv4_11_1x1_increase_bn_scale) + conv4_10_relu
conv4_11_relu = mx.symbol.Activation(name='conv4_11_relu', data=conv4_11 , act_type='relu')
conv4_12_1x1_reduce = mx.symbol.Convolution(name='conv4_12_1x1_reduce', data=conv4_11_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv4_12_1x1_reduce_bn = mx.symbol.BatchNorm(name='conv4_12_1x1_reduce_bn', data=conv4_12_1x1_reduce , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_12_1x1_reduce_bn_scale = conv4_12_1x1_reduce_bn
conv4_12_1x1_reduce_relu = mx.symbol.Activation(name='conv4_12_1x1_reduce_relu', data=conv4_12_1x1_reduce_bn_scale , act_type='relu')
conv4_12_3x3 = mx.symbol.Convolution(name='conv4_12_3x3', data=conv4_12_1x1_reduce_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
conv4_12_3x3_bn = mx.symbol.BatchNorm(name='conv4_12_3x3_bn', data=conv4_12_3x3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_12_3x3_bn_scale = conv4_12_3x3_bn
conv4_12_3x3_relu = mx.symbol.Activation(name='conv4_12_3x3_relu', data=conv4_12_3x3_bn_scale , act_type='relu')
conv4_12_1x1_increase = mx.symbol.Convolution(name='conv4_12_1x1_increase', data=conv4_12_3x3_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv4_12_1x1_increase_bn = mx.symbol.BatchNorm(name='conv4_12_1x1_increase_bn', data=conv4_12_1x1_increase , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_12_1x1_increase_bn_scale = conv4_12_1x1_increase_bn
conv4_12_global_pool = mx.symbol.Pooling(name='conv4_12_global_pool', data=conv4_12_1x1_increase_bn_scale , pooling_convention='full', global_pool=True, kernel=(1,1), pool_type='avg')
conv4_12_1x1_down = mx.symbol.Convolution(name='conv4_12_1x1_down', data=conv4_12_global_pool , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv4_12_1x1_down_relu = mx.symbol.Activation(name='conv4_12_1x1_down_relu', data=conv4_12_1x1_down , act_type='relu')
conv4_12_1x1_up = mx.symbol.Convolution(name='conv4_12_1x1_up', data=conv4_12_1x1_down_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv4_12_prob = mx.symbol.Activation(name='conv4_12_prob', data=conv4_12_1x1_up , act_type='sigmoid')
if memonger:
    conv4_11_relu._set_attr(mirror_stage='True')
conv4_12 = mx.sym.broadcast_mul(conv4_12_prob, conv4_12_1x1_increase_bn_scale) + conv4_11_relu
conv4_12_relu = mx.symbol.Activation(name='conv4_12_relu', data=conv4_12 , act_type='relu')
conv4_13_1x1_reduce = mx.symbol.Convolution(name='conv4_13_1x1_reduce', data=conv4_12_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv4_13_1x1_reduce_bn = mx.symbol.BatchNorm(name='conv4_13_1x1_reduce_bn', data=conv4_13_1x1_reduce , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_13_1x1_reduce_bn_scale = conv4_13_1x1_reduce_bn
conv4_13_1x1_reduce_relu = mx.symbol.Activation(name='conv4_13_1x1_reduce_relu', data=conv4_13_1x1_reduce_bn_scale , act_type='relu')
conv4_13_3x3 = mx.symbol.Convolution(name='conv4_13_3x3', data=conv4_13_1x1_reduce_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
conv4_13_3x3_bn = mx.symbol.BatchNorm(name='conv4_13_3x3_bn', data=conv4_13_3x3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_13_3x3_bn_scale = conv4_13_3x3_bn
conv4_13_3x3_relu = mx.symbol.Activation(name='conv4_13_3x3_relu', data=conv4_13_3x3_bn_scale , act_type='relu')
conv4_13_1x1_increase = mx.symbol.Convolution(name='conv4_13_1x1_increase', data=conv4_13_3x3_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv4_13_1x1_increase_bn = mx.symbol.BatchNorm(name='conv4_13_1x1_increase_bn', data=conv4_13_1x1_increase , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_13_1x1_increase_bn_scale = conv4_13_1x1_increase_bn
conv4_13_global_pool = mx.symbol.Pooling(name='conv4_13_global_pool', data=conv4_13_1x1_increase_bn_scale , pooling_convention='full', global_pool=True, kernel=(1,1), pool_type='avg')
conv4_13_1x1_down = mx.symbol.Convolution(name='conv4_13_1x1_down', data=conv4_13_global_pool , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv4_13_1x1_down_relu = mx.symbol.Activation(name='conv4_13_1x1_down_relu', data=conv4_13_1x1_down , act_type='relu')
conv4_13_1x1_up = mx.symbol.Convolution(name='conv4_13_1x1_up', data=conv4_13_1x1_down_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv4_13_prob = mx.symbol.Activation(name='conv4_13_prob', data=conv4_13_1x1_up , act_type='sigmoid')
if memonger:
    conv4_12_relu._set_attr(mirror_stage='True')
conv4_13 = mx.sym.broadcast_mul(conv4_13_prob, conv4_13_1x1_increase_bn_scale) + conv4_12_relu
conv4_13_relu = mx.symbol.Activation(name='conv4_13_relu', data=conv4_13 , act_type='relu')
conv4_14_1x1_reduce = mx.symbol.Convolution(name='conv4_14_1x1_reduce', data=conv4_13_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv4_14_1x1_reduce_bn = mx.symbol.BatchNorm(name='conv4_14_1x1_reduce_bn', data=conv4_14_1x1_reduce , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_14_1x1_reduce_bn_scale = conv4_14_1x1_reduce_bn
conv4_14_1x1_reduce_relu = mx.symbol.Activation(name='conv4_14_1x1_reduce_relu', data=conv4_14_1x1_reduce_bn_scale , act_type='relu')
conv4_14_3x3 = mx.symbol.Convolution(name='conv4_14_3x3', data=conv4_14_1x1_reduce_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
conv4_14_3x3_bn = mx.symbol.BatchNorm(name='conv4_14_3x3_bn', data=conv4_14_3x3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_14_3x3_bn_scale = conv4_14_3x3_bn
conv4_14_3x3_relu = mx.symbol.Activation(name='conv4_14_3x3_relu', data=conv4_14_3x3_bn_scale , act_type='relu')
conv4_14_1x1_increase = mx.symbol.Convolution(name='conv4_14_1x1_increase', data=conv4_14_3x3_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv4_14_1x1_increase_bn = mx.symbol.BatchNorm(name='conv4_14_1x1_increase_bn', data=conv4_14_1x1_increase , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_14_1x1_increase_bn_scale = conv4_14_1x1_increase_bn
conv4_14_global_pool = mx.symbol.Pooling(name='conv4_14_global_pool', data=conv4_14_1x1_increase_bn_scale , pooling_convention='full', global_pool=True, kernel=(1,1), pool_type='avg')
conv4_14_1x1_down = mx.symbol.Convolution(name='conv4_14_1x1_down', data=conv4_14_global_pool , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv4_14_1x1_down_relu = mx.symbol.Activation(name='conv4_14_1x1_down_relu', data=conv4_14_1x1_down , act_type='relu')
conv4_14_1x1_up = mx.symbol.Convolution(name='conv4_14_1x1_up', data=conv4_14_1x1_down_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv4_14_prob = mx.symbol.Activation(name='conv4_14_prob', data=conv4_14_1x1_up , act_type='sigmoid')
if memonger:
    conv4_13_relu._set_attr(mirror_stage='True')
conv4_14 = mx.sym.broadcast_mul(conv4_14_prob, conv4_14_1x1_increase_bn_scale) + conv4_13_relu
conv4_14_relu = mx.symbol.Activation(name='conv4_14_relu', data=conv4_14 , act_type='relu')
conv4_15_1x1_reduce = mx.symbol.Convolution(name='conv4_15_1x1_reduce', data=conv4_14_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv4_15_1x1_reduce_bn = mx.symbol.BatchNorm(name='conv4_15_1x1_reduce_bn', data=conv4_15_1x1_reduce , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_15_1x1_reduce_bn_scale = conv4_15_1x1_reduce_bn
conv4_15_1x1_reduce_relu = mx.symbol.Activation(name='conv4_15_1x1_reduce_relu', data=conv4_15_1x1_reduce_bn_scale , act_type='relu')
conv4_15_3x3 = mx.symbol.Convolution(name='conv4_15_3x3', data=conv4_15_1x1_reduce_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
conv4_15_3x3_bn = mx.symbol.BatchNorm(name='conv4_15_3x3_bn', data=conv4_15_3x3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_15_3x3_bn_scale = conv4_15_3x3_bn
conv4_15_3x3_relu = mx.symbol.Activation(name='conv4_15_3x3_relu', data=conv4_15_3x3_bn_scale , act_type='relu')
conv4_15_1x1_increase = mx.symbol.Convolution(name='conv4_15_1x1_increase', data=conv4_15_3x3_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv4_15_1x1_increase_bn = mx.symbol.BatchNorm(name='conv4_15_1x1_increase_bn', data=conv4_15_1x1_increase , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_15_1x1_increase_bn_scale = conv4_15_1x1_increase_bn
conv4_15_global_pool = mx.symbol.Pooling(name='conv4_15_global_pool', data=conv4_15_1x1_increase_bn_scale , pooling_convention='full', global_pool=True, kernel=(1,1), pool_type='avg')
conv4_15_1x1_down = mx.symbol.Convolution(name='conv4_15_1x1_down', data=conv4_15_global_pool , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv4_15_1x1_down_relu = mx.symbol.Activation(name='conv4_15_1x1_down_relu', data=conv4_15_1x1_down , act_type='relu')
conv4_15_1x1_up = mx.symbol.Convolution(name='conv4_15_1x1_up', data=conv4_15_1x1_down_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv4_15_prob = mx.symbol.Activation(name='conv4_15_prob', data=conv4_15_1x1_up , act_type='sigmoid')
if memonger:
    conv4_14_relu._set_attr(mirror_stage='True')
conv4_15 = mx.sym.broadcast_mul(conv4_15_prob, conv4_15_1x1_increase_bn_scale) + conv4_14_relu
conv4_15_relu = mx.symbol.Activation(name='conv4_15_relu', data=conv4_15 , act_type='relu')
conv4_16_1x1_reduce = mx.symbol.Convolution(name='conv4_16_1x1_reduce', data=conv4_15_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv4_16_1x1_reduce_bn = mx.symbol.BatchNorm(name='conv4_16_1x1_reduce_bn', data=conv4_16_1x1_reduce , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_16_1x1_reduce_bn_scale = conv4_16_1x1_reduce_bn
conv4_16_1x1_reduce_relu = mx.symbol.Activation(name='conv4_16_1x1_reduce_relu', data=conv4_16_1x1_reduce_bn_scale , act_type='relu')
conv4_16_3x3 = mx.symbol.Convolution(name='conv4_16_3x3', data=conv4_16_1x1_reduce_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
conv4_16_3x3_bn = mx.symbol.BatchNorm(name='conv4_16_3x3_bn', data=conv4_16_3x3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_16_3x3_bn_scale = conv4_16_3x3_bn
conv4_16_3x3_relu = mx.symbol.Activation(name='conv4_16_3x3_relu', data=conv4_16_3x3_bn_scale , act_type='relu')
conv4_16_1x1_increase = mx.symbol.Convolution(name='conv4_16_1x1_increase', data=conv4_16_3x3_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv4_16_1x1_increase_bn = mx.symbol.BatchNorm(name='conv4_16_1x1_increase_bn', data=conv4_16_1x1_increase , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_16_1x1_increase_bn_scale = conv4_16_1x1_increase_bn
conv4_16_global_pool = mx.symbol.Pooling(name='conv4_16_global_pool', data=conv4_16_1x1_increase_bn_scale , pooling_convention='full', global_pool=True, kernel=(1,1), pool_type='avg')
conv4_16_1x1_down = mx.symbol.Convolution(name='conv4_16_1x1_down', data=conv4_16_global_pool , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv4_16_1x1_down_relu = mx.symbol.Activation(name='conv4_16_1x1_down_relu', data=conv4_16_1x1_down , act_type='relu')
conv4_16_1x1_up = mx.symbol.Convolution(name='conv4_16_1x1_up', data=conv4_16_1x1_down_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv4_16_prob = mx.symbol.Activation(name='conv4_16_prob', data=conv4_16_1x1_up , act_type='sigmoid')
if memonger:
    conv4_15_relu._set_attr(mirror_stage='True')
conv4_16 = mx.sym.broadcast_mul(conv4_16_prob, conv4_16_1x1_increase_bn_scale) + conv4_15_relu
conv4_16_relu = mx.symbol.Activation(name='conv4_16_relu', data=conv4_16 , act_type='relu')
conv4_17_1x1_reduce = mx.symbol.Convolution(name='conv4_17_1x1_reduce', data=conv4_16_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv4_17_1x1_reduce_bn = mx.symbol.BatchNorm(name='conv4_17_1x1_reduce_bn', data=conv4_17_1x1_reduce , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_17_1x1_reduce_bn_scale = conv4_17_1x1_reduce_bn
conv4_17_1x1_reduce_relu = mx.symbol.Activation(name='conv4_17_1x1_reduce_relu', data=conv4_17_1x1_reduce_bn_scale , act_type='relu')
conv4_17_3x3 = mx.symbol.Convolution(name='conv4_17_3x3', data=conv4_17_1x1_reduce_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
conv4_17_3x3_bn = mx.symbol.BatchNorm(name='conv4_17_3x3_bn', data=conv4_17_3x3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_17_3x3_bn_scale = conv4_17_3x3_bn
conv4_17_3x3_relu = mx.symbol.Activation(name='conv4_17_3x3_relu', data=conv4_17_3x3_bn_scale , act_type='relu')
conv4_17_1x1_increase = mx.symbol.Convolution(name='conv4_17_1x1_increase', data=conv4_17_3x3_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv4_17_1x1_increase_bn = mx.symbol.BatchNorm(name='conv4_17_1x1_increase_bn', data=conv4_17_1x1_increase , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_17_1x1_increase_bn_scale = conv4_17_1x1_increase_bn
conv4_17_global_pool = mx.symbol.Pooling(name='conv4_17_global_pool', data=conv4_17_1x1_increase_bn_scale , pooling_convention='full', global_pool=True, kernel=(1,1), pool_type='avg')
conv4_17_1x1_down = mx.symbol.Convolution(name='conv4_17_1x1_down', data=conv4_17_global_pool , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv4_17_1x1_down_relu = mx.symbol.Activation(name='conv4_17_1x1_down_relu', data=conv4_17_1x1_down , act_type='relu')
conv4_17_1x1_up = mx.symbol.Convolution(name='conv4_17_1x1_up', data=conv4_17_1x1_down_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv4_17_prob = mx.symbol.Activation(name='conv4_17_prob', data=conv4_17_1x1_up , act_type='sigmoid')
if memonger:
    conv4_16_relu._set_attr(mirror_stage='True')
conv4_17 = mx.sym.broadcast_mul(conv4_17_prob, conv4_17_1x1_increase_bn_scale) + conv4_16_relu
conv4_17_relu = mx.symbol.Activation(name='conv4_17_relu', data=conv4_17 , act_type='relu')
conv4_18_1x1_reduce = mx.symbol.Convolution(name='conv4_18_1x1_reduce', data=conv4_17_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv4_18_1x1_reduce_bn = mx.symbol.BatchNorm(name='conv4_18_1x1_reduce_bn', data=conv4_18_1x1_reduce , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_18_1x1_reduce_bn_scale = conv4_18_1x1_reduce_bn
conv4_18_1x1_reduce_relu = mx.symbol.Activation(name='conv4_18_1x1_reduce_relu', data=conv4_18_1x1_reduce_bn_scale , act_type='relu')
conv4_18_3x3 = mx.symbol.Convolution(name='conv4_18_3x3', data=conv4_18_1x1_reduce_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
conv4_18_3x3_bn = mx.symbol.BatchNorm(name='conv4_18_3x3_bn', data=conv4_18_3x3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_18_3x3_bn_scale = conv4_18_3x3_bn
conv4_18_3x3_relu = mx.symbol.Activation(name='conv4_18_3x3_relu', data=conv4_18_3x3_bn_scale , act_type='relu')
conv4_18_1x1_increase = mx.symbol.Convolution(name='conv4_18_1x1_increase', data=conv4_18_3x3_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv4_18_1x1_increase_bn = mx.symbol.BatchNorm(name='conv4_18_1x1_increase_bn', data=conv4_18_1x1_increase , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_18_1x1_increase_bn_scale = conv4_18_1x1_increase_bn
conv4_18_global_pool = mx.symbol.Pooling(name='conv4_18_global_pool', data=conv4_18_1x1_increase_bn_scale , pooling_convention='full', global_pool=True, kernel=(1,1), pool_type='avg')
conv4_18_1x1_down = mx.symbol.Convolution(name='conv4_18_1x1_down', data=conv4_18_global_pool , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv4_18_1x1_down_relu = mx.symbol.Activation(name='conv4_18_1x1_down_relu', data=conv4_18_1x1_down , act_type='relu')
conv4_18_1x1_up = mx.symbol.Convolution(name='conv4_18_1x1_up', data=conv4_18_1x1_down_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv4_18_prob = mx.symbol.Activation(name='conv4_18_prob', data=conv4_18_1x1_up , act_type='sigmoid')
if memonger:
    conv4_17_relu._set_attr(mirror_stage='True')
conv4_18 = mx.sym.broadcast_mul(conv4_18_prob, conv4_18_1x1_increase_bn_scale) + conv4_17_relu
conv4_18_relu = mx.symbol.Activation(name='conv4_18_relu', data=conv4_18 , act_type='relu')
conv4_19_1x1_reduce = mx.symbol.Convolution(name='conv4_19_1x1_reduce', data=conv4_18_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv4_19_1x1_reduce_bn = mx.symbol.BatchNorm(name='conv4_19_1x1_reduce_bn', data=conv4_19_1x1_reduce , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_19_1x1_reduce_bn_scale = conv4_19_1x1_reduce_bn
conv4_19_1x1_reduce_relu = mx.symbol.Activation(name='conv4_19_1x1_reduce_relu', data=conv4_19_1x1_reduce_bn_scale , act_type='relu')
conv4_19_3x3 = mx.symbol.Convolution(name='conv4_19_3x3', data=conv4_19_1x1_reduce_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
conv4_19_3x3_bn = mx.symbol.BatchNorm(name='conv4_19_3x3_bn', data=conv4_19_3x3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_19_3x3_bn_scale = conv4_19_3x3_bn
conv4_19_3x3_relu = mx.symbol.Activation(name='conv4_19_3x3_relu', data=conv4_19_3x3_bn_scale , act_type='relu')
conv4_19_1x1_increase = mx.symbol.Convolution(name='conv4_19_1x1_increase', data=conv4_19_3x3_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv4_19_1x1_increase_bn = mx.symbol.BatchNorm(name='conv4_19_1x1_increase_bn', data=conv4_19_1x1_increase , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_19_1x1_increase_bn_scale = conv4_19_1x1_increase_bn
conv4_19_global_pool = mx.symbol.Pooling(name='conv4_19_global_pool', data=conv4_19_1x1_increase_bn_scale , pooling_convention='full', global_pool=True, kernel=(1,1), pool_type='avg')
conv4_19_1x1_down = mx.symbol.Convolution(name='conv4_19_1x1_down', data=conv4_19_global_pool , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv4_19_1x1_down_relu = mx.symbol.Activation(name='conv4_19_1x1_down_relu', data=conv4_19_1x1_down , act_type='relu')
conv4_19_1x1_up = mx.symbol.Convolution(name='conv4_19_1x1_up', data=conv4_19_1x1_down_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv4_19_prob = mx.symbol.Activation(name='conv4_19_prob', data=conv4_19_1x1_up , act_type='sigmoid')
if memonger:
    conv4_18_relu._set_attr(mirror_stage='True')
conv4_19 = mx.sym.broadcast_mul(conv4_19_prob, conv4_19_1x1_increase_bn_scale) + conv4_18_relu
conv4_19_relu = mx.symbol.Activation(name='conv4_19_relu', data=conv4_19 , act_type='relu')
conv4_20_1x1_reduce = mx.symbol.Convolution(name='conv4_20_1x1_reduce', data=conv4_19_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv4_20_1x1_reduce_bn = mx.symbol.BatchNorm(name='conv4_20_1x1_reduce_bn', data=conv4_20_1x1_reduce , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_20_1x1_reduce_bn_scale = conv4_20_1x1_reduce_bn
conv4_20_1x1_reduce_relu = mx.symbol.Activation(name='conv4_20_1x1_reduce_relu', data=conv4_20_1x1_reduce_bn_scale , act_type='relu')
conv4_20_3x3 = mx.symbol.Convolution(name='conv4_20_3x3', data=conv4_20_1x1_reduce_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
conv4_20_3x3_bn = mx.symbol.BatchNorm(name='conv4_20_3x3_bn', data=conv4_20_3x3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_20_3x3_bn_scale = conv4_20_3x3_bn
conv4_20_3x3_relu = mx.symbol.Activation(name='conv4_20_3x3_relu', data=conv4_20_3x3_bn_scale , act_type='relu')
conv4_20_1x1_increase = mx.symbol.Convolution(name='conv4_20_1x1_increase', data=conv4_20_3x3_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv4_20_1x1_increase_bn = mx.symbol.BatchNorm(name='conv4_20_1x1_increase_bn', data=conv4_20_1x1_increase , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_20_1x1_increase_bn_scale = conv4_20_1x1_increase_bn
conv4_20_global_pool = mx.symbol.Pooling(name='conv4_20_global_pool', data=conv4_20_1x1_increase_bn_scale , pooling_convention='full', global_pool=True, kernel=(1,1), pool_type='avg')
conv4_20_1x1_down = mx.symbol.Convolution(name='conv4_20_1x1_down', data=conv4_20_global_pool , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv4_20_1x1_down_relu = mx.symbol.Activation(name='conv4_20_1x1_down_relu', data=conv4_20_1x1_down , act_type='relu')
conv4_20_1x1_up = mx.symbol.Convolution(name='conv4_20_1x1_up', data=conv4_20_1x1_down_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv4_20_prob = mx.symbol.Activation(name='conv4_20_prob', data=conv4_20_1x1_up , act_type='sigmoid')
if memonger:
    conv4_19_relu._set_attr(mirror_stage='True')
conv4_20 = mx.sym.broadcast_mul(conv4_20_prob, conv4_20_1x1_increase_bn_scale) + conv4_19_relu
conv4_20_relu = mx.symbol.Activation(name='conv4_20_relu', data=conv4_20 , act_type='relu')
conv4_21_1x1_reduce = mx.symbol.Convolution(name='conv4_21_1x1_reduce', data=conv4_20_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv4_21_1x1_reduce_bn = mx.symbol.BatchNorm(name='conv4_21_1x1_reduce_bn', data=conv4_21_1x1_reduce , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_21_1x1_reduce_bn_scale = conv4_21_1x1_reduce_bn
conv4_21_1x1_reduce_relu = mx.symbol.Activation(name='conv4_21_1x1_reduce_relu', data=conv4_21_1x1_reduce_bn_scale , act_type='relu')
conv4_21_3x3 = mx.symbol.Convolution(name='conv4_21_3x3', data=conv4_21_1x1_reduce_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
conv4_21_3x3_bn = mx.symbol.BatchNorm(name='conv4_21_3x3_bn', data=conv4_21_3x3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_21_3x3_bn_scale = conv4_21_3x3_bn
conv4_21_3x3_relu = mx.symbol.Activation(name='conv4_21_3x3_relu', data=conv4_21_3x3_bn_scale , act_type='relu')
conv4_21_1x1_increase = mx.symbol.Convolution(name='conv4_21_1x1_increase', data=conv4_21_3x3_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv4_21_1x1_increase_bn = mx.symbol.BatchNorm(name='conv4_21_1x1_increase_bn', data=conv4_21_1x1_increase , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_21_1x1_increase_bn_scale = conv4_21_1x1_increase_bn
conv4_21_global_pool = mx.symbol.Pooling(name='conv4_21_global_pool', data=conv4_21_1x1_increase_bn_scale , pooling_convention='full', global_pool=True, kernel=(1,1), pool_type='avg')
conv4_21_1x1_down = mx.symbol.Convolution(name='conv4_21_1x1_down', data=conv4_21_global_pool , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv4_21_1x1_down_relu = mx.symbol.Activation(name='conv4_21_1x1_down_relu', data=conv4_21_1x1_down , act_type='relu')
conv4_21_1x1_up = mx.symbol.Convolution(name='conv4_21_1x1_up', data=conv4_21_1x1_down_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv4_21_prob = mx.symbol.Activation(name='conv4_21_prob', data=conv4_21_1x1_up , act_type='sigmoid')
if memonger:
    conv4_20_relu._set_attr(mirror_stage='True')
conv4_21 = mx.sym.broadcast_mul(conv4_21_prob, conv4_21_1x1_increase_bn_scale) + conv4_20_relu
conv4_21_relu = mx.symbol.Activation(name='conv4_21_relu', data=conv4_21 , act_type='relu')
conv4_22_1x1_reduce = mx.symbol.Convolution(name='conv4_22_1x1_reduce', data=conv4_21_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv4_22_1x1_reduce_bn = mx.symbol.BatchNorm(name='conv4_22_1x1_reduce_bn', data=conv4_22_1x1_reduce , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_22_1x1_reduce_bn_scale = conv4_22_1x1_reduce_bn
conv4_22_1x1_reduce_relu = mx.symbol.Activation(name='conv4_22_1x1_reduce_relu', data=conv4_22_1x1_reduce_bn_scale , act_type='relu')
conv4_22_3x3 = mx.symbol.Convolution(name='conv4_22_3x3', data=conv4_22_1x1_reduce_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
conv4_22_3x3_bn = mx.symbol.BatchNorm(name='conv4_22_3x3_bn', data=conv4_22_3x3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_22_3x3_bn_scale = conv4_22_3x3_bn
conv4_22_3x3_relu = mx.symbol.Activation(name='conv4_22_3x3_relu', data=conv4_22_3x3_bn_scale , act_type='relu')
conv4_22_1x1_increase = mx.symbol.Convolution(name='conv4_22_1x1_increase', data=conv4_22_3x3_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv4_22_1x1_increase_bn = mx.symbol.BatchNorm(name='conv4_22_1x1_increase_bn', data=conv4_22_1x1_increase , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_22_1x1_increase_bn_scale = conv4_22_1x1_increase_bn
conv4_22_global_pool = mx.symbol.Pooling(name='conv4_22_global_pool', data=conv4_22_1x1_increase_bn_scale , pooling_convention='full', global_pool=True, kernel=(1,1), pool_type='avg')
conv4_22_1x1_down = mx.symbol.Convolution(name='conv4_22_1x1_down', data=conv4_22_global_pool , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv4_22_1x1_down_relu = mx.symbol.Activation(name='conv4_22_1x1_down_relu', data=conv4_22_1x1_down , act_type='relu')
conv4_22_1x1_up = mx.symbol.Convolution(name='conv4_22_1x1_up', data=conv4_22_1x1_down_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv4_22_prob = mx.symbol.Activation(name='conv4_22_prob', data=conv4_22_1x1_up , act_type='sigmoid')
if memonger:
    conv4_21_relu._set_attr(mirror_stage='True')
conv4_22 = mx.sym.broadcast_mul(conv4_22_prob, conv4_22_1x1_increase_bn_scale) + conv4_21_relu
conv4_22_relu = mx.symbol.Activation(name='conv4_22_relu', data=conv4_22 , act_type='relu')
conv4_23_1x1_reduce = mx.symbol.Convolution(name='conv4_23_1x1_reduce', data=conv4_22_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv4_23_1x1_reduce_bn = mx.symbol.BatchNorm(name='conv4_23_1x1_reduce_bn', data=conv4_23_1x1_reduce , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_23_1x1_reduce_bn_scale = conv4_23_1x1_reduce_bn
conv4_23_1x1_reduce_relu = mx.symbol.Activation(name='conv4_23_1x1_reduce_relu', data=conv4_23_1x1_reduce_bn_scale , act_type='relu')
conv4_23_3x3 = mx.symbol.Convolution(name='conv4_23_3x3', data=conv4_23_1x1_reduce_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
conv4_23_3x3_bn = mx.symbol.BatchNorm(name='conv4_23_3x3_bn', data=conv4_23_3x3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_23_3x3_bn_scale = conv4_23_3x3_bn
conv4_23_3x3_relu = mx.symbol.Activation(name='conv4_23_3x3_relu', data=conv4_23_3x3_bn_scale , act_type='relu')
conv4_23_1x1_increase = mx.symbol.Convolution(name='conv4_23_1x1_increase', data=conv4_23_3x3_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv4_23_1x1_increase_bn = mx.symbol.BatchNorm(name='conv4_23_1x1_increase_bn', data=conv4_23_1x1_increase , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv4_23_1x1_increase_bn_scale = conv4_23_1x1_increase_bn
conv4_23_global_pool = mx.symbol.Pooling(name='conv4_23_global_pool', data=conv4_23_1x1_increase_bn_scale , pooling_convention='full', global_pool=True, kernel=(1,1), pool_type='avg')
conv4_23_1x1_down = mx.symbol.Convolution(name='conv4_23_1x1_down', data=conv4_23_global_pool , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv4_23_1x1_down_relu = mx.symbol.Activation(name='conv4_23_1x1_down_relu', data=conv4_23_1x1_down , act_type='relu')
conv4_23_1x1_up = mx.symbol.Convolution(name='conv4_23_1x1_up', data=conv4_23_1x1_down_relu , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv4_23_prob = mx.symbol.Activation(name='conv4_23_prob', data=conv4_23_1x1_up , act_type='sigmoid')
if memonger:
    conv4_22_relu._set_attr(mirror_stage='True')
conv4_23 = mx.sym.broadcast_mul(conv4_23_prob, conv4_23_1x1_increase_bn_scale) + conv4_22_relu
conv4_23_relu = mx.symbol.Activation(name='conv4_23_relu', data=conv4_23 , act_type='relu')
conv5_1_1x1_reduce = mx.symbol.Convolution(name='conv5_1_1x1_reduce', data=conv4_23_relu , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
conv5_1_1x1_reduce_bn = mx.symbol.BatchNorm(name='conv5_1_1x1_reduce_bn', data=conv5_1_1x1_reduce , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv5_1_1x1_reduce_bn_scale = conv5_1_1x1_reduce_bn
conv5_1_1x1_reduce_relu = mx.symbol.Activation(name='conv5_1_1x1_reduce_relu', data=conv5_1_1x1_reduce_bn_scale , act_type='relu')
conv5_1_3x3 = mx.symbol.Convolution(name='conv5_1_3x3', data=conv5_1_1x1_reduce_relu , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
conv5_1_3x3_bn = mx.symbol.BatchNorm(name='conv5_1_3x3_bn', data=conv5_1_3x3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv5_1_3x3_bn_scale = conv5_1_3x3_bn
conv5_1_3x3_relu = mx.symbol.Activation(name='conv5_1_3x3_relu', data=conv5_1_3x3_bn_scale , act_type='relu')
conv5_1_1x1_increase = mx.symbol.Convolution(name='conv5_1_1x1_increase', data=conv5_1_3x3_relu , num_filter=2048, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv5_1_1x1_increase_bn = mx.symbol.BatchNorm(name='conv5_1_1x1_increase_bn', data=conv5_1_1x1_increase , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv5_1_1x1_increase_bn_scale = conv5_1_1x1_increase_bn
conv5_1_global_pool = mx.symbol.Pooling(name='conv5_1_global_pool', data=conv5_1_1x1_increase_bn_scale , pooling_convention='full', global_pool=True, kernel=(1,1), pool_type='avg')
conv5_1_1x1_down = mx.symbol.Convolution(name='conv5_1_1x1_down', data=conv5_1_global_pool , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv5_1_1x1_down_relu = mx.symbol.Activation(name='conv5_1_1x1_down_relu', data=conv5_1_1x1_down , act_type='relu')
conv5_1_1x1_up = mx.symbol.Convolution(name='conv5_1_1x1_up', data=conv5_1_1x1_down_relu , num_filter=2048, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv5_1_prob = mx.symbol.Activation(name='conv5_1_prob', data=conv5_1_1x1_up , act_type='sigmoid')
conv5_1_1x1_proj = mx.symbol.Convolution(name='conv5_1_1x1_proj', data=conv4_23_relu , num_filter=2048, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
conv5_1_1x1_proj_bn = mx.symbol.BatchNorm(name='conv5_1_1x1_proj_bn', data=conv5_1_1x1_proj , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv5_1_1x1_proj_bn_scale = conv5_1_1x1_proj_bn
if memonger:
    conv5_1_1x1_proj_bn_scale._set_attr(mirror_stage='True')
conv5_1 = mx.sym.broadcast_mul(conv5_1_prob, conv5_1_1x1_increase_bn_scale) + conv5_1_1x1_proj_bn_scale
conv5_1_relu = mx.symbol.Activation(name='conv5_1_relu', data=conv5_1 , act_type='relu')
conv5_2_1x1_reduce = mx.symbol.Convolution(name='conv5_2_1x1_reduce', data=conv5_1_relu , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv5_2_1x1_reduce_bn = mx.symbol.BatchNorm(name='conv5_2_1x1_reduce_bn', data=conv5_2_1x1_reduce , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv5_2_1x1_reduce_bn_scale = conv5_2_1x1_reduce_bn
conv5_2_1x1_reduce_relu = mx.symbol.Activation(name='conv5_2_1x1_reduce_relu', data=conv5_2_1x1_reduce_bn_scale , act_type='relu')
conv5_2_3x3 = mx.symbol.Convolution(name='conv5_2_3x3', data=conv5_2_1x1_reduce_relu , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
conv5_2_3x3_bn = mx.symbol.BatchNorm(name='conv5_2_3x3_bn', data=conv5_2_3x3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv5_2_3x3_bn_scale = conv5_2_3x3_bn
conv5_2_3x3_relu = mx.symbol.Activation(name='conv5_2_3x3_relu', data=conv5_2_3x3_bn_scale , act_type='relu')
conv5_2_1x1_increase = mx.symbol.Convolution(name='conv5_2_1x1_increase', data=conv5_2_3x3_relu , num_filter=2048, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv5_2_1x1_increase_bn = mx.symbol.BatchNorm(name='conv5_2_1x1_increase_bn', data=conv5_2_1x1_increase , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv5_2_1x1_increase_bn_scale = conv5_2_1x1_increase_bn
conv5_2_global_pool = mx.symbol.Pooling(name='conv5_2_global_pool', data=conv5_2_1x1_increase_bn_scale , pooling_convention='full', global_pool=True, kernel=(1,1), pool_type='avg')
conv5_2_1x1_down = mx.symbol.Convolution(name='conv5_2_1x1_down', data=conv5_2_global_pool , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv5_2_1x1_down_relu = mx.symbol.Activation(name='conv5_2_1x1_down_relu', data=conv5_2_1x1_down , act_type='relu')
conv5_2_1x1_up = mx.symbol.Convolution(name='conv5_2_1x1_up', data=conv5_2_1x1_down_relu , num_filter=2048, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv5_2_prob = mx.symbol.Activation(name='conv5_2_prob', data=conv5_2_1x1_up , act_type='sigmoid')
if memonger:
    conv5_1_relu._set_attr(mirror_stage='True')
conv5_2 = mx.sym.broadcast_mul(conv5_2_prob, conv5_2_1x1_increase_bn_scale) + conv5_1_relu
conv5_2_relu = mx.symbol.Activation(name='conv5_2_relu', data=conv5_2 , act_type='relu')
conv5_3_1x1_reduce = mx.symbol.Convolution(name='conv5_3_1x1_reduce', data=conv5_2_relu , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv5_3_1x1_reduce_bn = mx.symbol.BatchNorm(name='conv5_3_1x1_reduce_bn', data=conv5_3_1x1_reduce , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv5_3_1x1_reduce_bn_scale = conv5_3_1x1_reduce_bn
conv5_3_1x1_reduce_relu = mx.symbol.Activation(name='conv5_3_1x1_reduce_relu', data=conv5_3_1x1_reduce_bn_scale , act_type='relu')
conv5_3_3x3 = mx.symbol.Convolution(name='conv5_3_3x3', data=conv5_3_1x1_reduce_relu , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
conv5_3_3x3_bn = mx.symbol.BatchNorm(name='conv5_3_3x3_bn', data=conv5_3_3x3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv5_3_3x3_bn_scale = conv5_3_3x3_bn
conv5_3_3x3_relu = mx.symbol.Activation(name='conv5_3_3x3_relu', data=conv5_3_3x3_bn_scale , act_type='relu')
conv5_3_1x1_increase = mx.symbol.Convolution(name='conv5_3_1x1_increase', data=conv5_3_3x3_relu , num_filter=2048, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv5_3_1x1_increase_bn = mx.symbol.BatchNorm(name='conv5_3_1x1_increase_bn', data=conv5_3_1x1_increase , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv5_3_1x1_increase_bn_scale = conv5_3_1x1_increase_bn
conv5_3_global_pool = mx.symbol.Pooling(name='conv5_3_global_pool', data=conv5_3_1x1_increase_bn_scale , pooling_convention='full', global_pool=True, kernel=(1,1), pool_type='avg')
conv5_3_1x1_down = mx.symbol.Convolution(name='conv5_3_1x1_down', data=conv5_3_global_pool , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv5_3_1x1_down_relu = mx.symbol.Activation(name='conv5_3_1x1_down_relu', data=conv5_3_1x1_down , act_type='relu')
conv5_3_1x1_up = mx.symbol.Convolution(name='conv5_3_1x1_up', data=conv5_3_1x1_down_relu , num_filter=2048, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
conv5_3_prob = mx.symbol.Activation(name='conv5_3_prob', data=conv5_3_1x1_up , act_type='sigmoid')
if memonger:
    conv5_2_relu._set_attr(mirror_stage='True')
conv5_3 = mx.sym.broadcast_mul(conv5_3_prob, conv5_3_1x1_increase_bn_scale) + conv5_2_relu
conv5_3_relu = mx.symbol.Activation(name='conv5_3_relu', data=conv5_3 , act_type='relu')
pool5_7x7_s1 = mx.symbol.Pooling(name='pool5_7x7_s1', data=conv5_3_relu , pooling_convention='full', pad=(0,0), kernel=(7,7), stride=(1,1), pool_type='avg')
flatten_0=mx.symbol.Flatten(name='flatten_0', data=pool5_7x7_s1)
classifier = mx.symbol.FullyConnected(name='classifier', data=flatten_0 , num_hidden=1000, no_bias=False)
prob = mx.symbol.SoftmaxOutput(name='prob', data=classifier )
