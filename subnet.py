"""
Define the network for retraining

"""
import random
import mxnet as mx
import mxnet
import numpy as np
from mxnet.gluon import nn
from mxnet.gluon.block import HybridBlock
from blocks import Shufflenet, Shuffle_Xception, Activation
from mxnet import ndarray as F


class ShuffleNetV2_OneShot(HybridBlock):
    def __init__(self, input_size=224, n_class=1000, architecture=None, channels_idx=None, act_type='relu', search=False):
        super(ShuffleNetV2_OneShot, self).__init__()

        assert input_size % 32 == 0
        assert architecture is not None and channels_idx is not None
        self.stage_repeats = [4, 4, 8, 4]
        self.stage_out_channels = [-1, 16, 64, 160, 320, 640, 1024]
        self.candidate_scales = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        #self.stage_out_channels = [-1, 16, 48, 128, 256, 512, 1024]
        input_channel = self.stage_out_channels[1]
        self.search = search

        self.first_conv = nn.HybridSequential(prefix='first_')
        self.first_conv.add(nn.Conv2D(input_channel, in_channels=3, kernel_size=3, strides=2, padding=1, use_bias=False))
        self.first_conv.add(nn.BatchNorm(in_channels=input_channel, momentum=0.1))
        self.first_conv.add(Activation(act_type))

        self.features = nn.HybridSequential(prefix='features_')
        archIndex = 0
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]

            for i in range(numrepeat):
                if i == 0:
                    inp, outp, stride = input_channel, output_channel, 2
                else:
                    inp, outp, stride = input_channel, output_channel, 1

                blockIndex = architecture[archIndex]
                base_mid_channels = outp // 2
                mid_channels = int(base_mid_channels * self.candidate_scales[channels_idx[archIndex]])
                archIndex += 1
                self.features.add(nn.HybridSequential(prefix=''))

                if blockIndex == 0:
                    #print('Shuffle3x3')
                    self.features[-1].add(Shufflenet(inp, outp, mid_channels=mid_channels, ksize=3, stride=stride,
                                                     act_type='relu', BatchNorm=nn.BatchNorm, search=self.search))
                elif blockIndex == 1:
                    #print('Shuffle5x5')
                    self.features[-1].add(Shufflenet(inp, outp, mid_channels=mid_channels, ksize=5, stride=stride,
                                                     act_type='relu', BatchNorm=nn.BatchNorm, search=self.search))
                elif blockIndex == 2:
                    #print('Shuffle7x7')
                    self.features[-1].add(Shufflenet(inp, outp, mid_channels=mid_channels, ksize=7, stride=stride,
                                                     act_type='relu', BatchNorm=nn.BatchNorm, search=self.search))
                elif blockIndex == 3:
                    #print('Xception')
                    self.features[-1].add(Shuffle_Xception(inp, outp, mid_channels=mid_channels, stride=stride,
                                                           act_type='relu', BatchNorm=nn.BatchNorm, search=self.search))
                else:
                    raise NotImplementedError
                input_channel = output_channel
        assert archIndex == len(architecture)
        self.conv_last = nn.HybridSequential(prefix='last_')
        self.conv_last.add(nn.Conv2D(self.stage_out_channels[-1], in_channels=input_channel, kernel_size=1, strides=1, padding=0, use_bias=False))
        self.conv_last.add(nn.BatchNorm(in_channels=self.stage_out_channels[-1],momentum=0.1))
        self.conv_last.add(Activation(act_type))

        self.globalpool = nn.GlobalAvgPool2D()
        self.output = nn.HybridSequential(prefix='output_')
        with self.output.name_scope():
            self.output.add(
                nn.Dropout(0.1),
                nn.Dense(units=n_class, in_units=self.stage_out_channels[-1], use_bias=False)

            )
        #self._initialize()

    def _initialize(self, force_reinit=True, ctx=mx.cpu()):
        for k, v in self.collect_params().items():
            if 'conv' in k:
                if 'weight' in k:
                    if 'first' in k:
                        v.initialize(mx.init.Normal(0.01), force_reinit=force_reinit, ctx=ctx)
                    else:
                        v.initialize(mx.init.Normal(1.0 / v.shape[1]), force_reinit=force_reinit, ctx=ctx)
                if 'bias' in k:
                    v.initialize(mx.init.Constant(0), force_reinit=force_reinit, ctx=ctx)
            elif 'batchnorm' in k:
                if 'gamma' in k:
                    v.initialize(mx.init.Constant(1), force_reinit=force_reinit, ctx=ctx)
                if 'beta' in k:
                    v.initialize(mx.init.Constant(0.0001), force_reinit=force_reinit, ctx=ctx)
                if 'running' in k or 'moving' in k:
                    v.initialize(mx.init.Constant(0), force_reinit=force_reinit, ctx=ctx)
            elif 'dense' in k:
                v.initialize(mx.init.Normal(0.01), force_reinit=force_reinit, ctx=ctx)
                if 'bias' in k:
                    v.initialize(mx.init.Constant(0), force_reinit=force_reinit, ctx=ctx)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.first_conv(x)
        x = self.features(x)
        x = self.conv_last(x)
        x = self.globalpool(x)
        x = self.output(x)
        return x


if __name__ == '__main__':
    can =(0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2)
    scale_ids = [6, 5, 3, 5, 2, 6, 3, 4, 2, 5, 7, 5, 4, 6, 7, 4, 4, 5, 4, 3]
    model = ShuffleNetV2_OneShot(input_size=224, n_class=1000, architecture=can, channels_idx=(4,)*20, act_type='relu', search=False)# define a specific subnet
    model.hybridize()
    model._initialize(ctx=mxnet.cpu())

    print(model)
    test_data = mxnet.nd.random.uniform(-1, 1, shape=(5, 3, 224, 224))
    test_outputs = model(test_data)
    #model.export('model')
    print(test_outputs)
    print(test_outputs.shape)
