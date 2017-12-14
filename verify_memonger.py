import sys
sys.path.insert(0, '/data2/obj_detect/mxnet/0.11.0/python')
from memonger import memonger
from SE_ResNeXt_101 import get_symbol

net = get_symbol(num_classes=1000, memonger=True, use_global_stats=True)

network = memonger.search_plan(net, data=(16,3,224,224))


