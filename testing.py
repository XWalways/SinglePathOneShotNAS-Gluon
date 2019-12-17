'''
from mxnet import nd
from mxnet import ndarray as F
channel_mask = []
L = [4,8,4,4]
stage_out_channels = [64, 160, 320, 640]
candidate_scales = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
channel_choice = [6, 5, 3, 5, 2, 6, 3, 4, 2, 5, 7, 5, 4, 6, 7, 4, 4, 5, 4, 3]
global_max_length = 640 #int(self.net.stage_out_channels[-1] // 2 * self.net.candidate_scales[-1])
for i in range(len(L)):
    for j in range(L[i]):
        local_mask = [0] * global_max_length
        channel_choice_index = len(channel_mask)  # channel_choice index is equal to current channel_mask length
        channel_num = int(stage_out_channels[i] // 2 *
                          candidate_scales[channel_choice[channel_choice_index]])
        local_mask[:channel_num] = [1] * channel_num
        channel_mask.append(local_mask)
channel_mask = nd.array(channel_mask).astype(dtype='float32', copy=False)
#print(len(channel_mask))
#for i in range(len(channel_mask)):
#    print(channel_mask[i])
print(channel_mask[0].shape)
#block_channel_mask = F.slice(channel_mask, begin=(0, None), end=(1, None))
block_channel_mask = F.slice(channel_mask[0], begin=(None,), end=(64,))
block_channel_mask = F.reshape(block_channel_mask, shape=(1, 64, 1, 1))
print(block_channel_mask)
'''
'''
import mxnet
from mxnet.gluon.nn import HybridBlock
from mxnet.gluon import nn
class testnet(HybridBlock):
    def __init__(self):
        super(testnet, self).__init__()
        self.convs = nn.HybridSequential(prefix='')
        self.convs.add(nn.Conv2D(16, in_channels=3, kernel_size=3, strides=1, padding=1, use_bias=False))
        self.convs.add(nn.Conv2D(16, in_channels=16, kernel_size=3, strides=1, padding=1, use_bias=False))
    def hybrid_forward(self, F, x):
        x = self.convs[0](x)
        x = self.convs[1](x)
        return x


model = testnet()
model.hybridize()
model.initialize()
test_data = mxnet.nd.random.uniform(-1, 1, shape=(5, 3, 32, 32))
output = model(test_data)
model.export('test')
for k, v in model.collect_params().items():
    print(k)
'''

import mxnet
import network
architecture = [0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2]
architecture = mxnet.nd.array(architecture).astype(dtype='float32', copy=False)
channel_choice = [6, 5, 3, 5, 2, 6, 3, 4, 2, 5, 7, 5, 4, 6, 7, 4, 4, 5, 4, 3]
channel_mask = network.get_channel_mask(channel_choice, dtype='float32')

test_model = network.ShuffleNetV2_OneShot(use_all_blocks=False, search=True)
#test_model._initialize(ctx=mxnet.cpu())
test_data = mxnet.nd.random.uniform(0, 0, shape=(5, 3, 224, 224))

test_model.collect_params().load('supernet.params')
for k, v in test_model.collect_params().items():
    if 'weight' in k:
        print(k)
o = test_model(test_data, architecture, channel_mask)
print(o)