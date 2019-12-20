from mxnet.gluon import nn
from mxnet.gluon.block import HybridBlock
from mxnet import nd
import random
import mxnet as mx
import mxnet
import numpy as np
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
        input_channel = self.stage_out_channels[1]

        self.first_conv = nn.HybridSequential(prefix='first_')
        self.first_conv.add(nn.Conv2D(input_channel, in_channels=3, kernel_size=3, strides=2, padding=1, use_bias=False))
        self.first_conv.add(nn.BatchNorm(in_channels=input_channel, momentum=0.1))
        self.first_conv.add(Activation(act_type))
        self.search = search

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
        self._initialize()

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

def get_flops(model):
    '''
    # use the package mxop
    inputs = nd.random.uniform(-1, 1, shape=(1, 3, 224, 224), ctx=mx.cpu(0))
    from mxop.gluon import count_ops
    op_counter = count_ops(model, input_size=(1,3,224,224))
    return op_counter
    '''

    list_conv = []
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].shape
        output_channels, output_height, output_width = output[0].shape
        assert self._in_channels % self._kwargs['num_group'] == 0

        kernel_ops = self._kwargs['kernel'][0] * self._kwargs['kernel'][1] * (self._in_channels // self._kwargs['num_group'])
        params = output_channels * kernel_ops
        flops = batch_size * params * output_height * output_width

        list_conv.append(flops)

    list_dense = []
    def dense_hook(self, input, output):
        batch_size = input[0].shape[0] if len(input[0].shape) == 2 else 1

        weight_ops = self.weight.shape[0] * self.weight.shape[1]
        #print(self.weight.shape)

        flops = batch_size * weight_ops
        list_dense.append(flops)

    def get(net):
        for op in net.first_conv:
            if isinstance(op, nn.Conv2D):
                op.register_forward_hook(conv_hook)
            if isinstance(op, nn.Dense):
                op.register_forward_hook(dense_hook)
        for blocks in net.features:
            for block in blocks:
                if hasattr(block, 'branch_proj'):
                    for op in block.branch_proj:
                        if isinstance(op, nn.Conv2D):
                            op.register_forward_hook(conv_hook)
                        if isinstance(op, nn.Dense):
                            op.register_forward_hook(dense_hook)
                for op in block.branch_main:
                    if isinstance(op, nn.Conv2D):
                        op.register_forward_hook(conv_hook)
                    if isinstance(op, nn.Dense):
                        op.register_forward_hook(dense_hook)
                    if isinstance(op, nn.HybridSequential):
                        for OP in op:
                            if isinstance(OP, nn.Conv2D):
                                OP.register_forward_hook(conv_hook)
                            if isinstance(OP, nn.Dense):
                                OP.register_forward_hook(dense_hook)
        for op in net.conv_last:
            if isinstance(op, nn.Conv2D):
                op.register_forward_hook(conv_hook)
            if isinstance(op, nn.Dense):
                op.register_forward_hook(dense_hook)
        for op in net.output:
            if isinstance(op, nn.Conv2D):
                op.register_forward_hook(conv_hook)
            if isinstance(op, nn.Dense):
                op.register_forward_hook(dense_hook)
    get(model)
    input = nd.random.uniform(-1, 1, shape=(1, 3, 224, 224), ctx=mx.cpu(0))
    out = model(input)
    total_flops = sum(sum(i) for i in [list_conv, list_dense])
    return total_flops


def get_cand_flops(cand, channels_idx):
    model = ShuffleNetV2_OneShot(input_size=224, n_class=1000, architecture=cand, channels_idx=channels_idx, act_type='relu', search=False)
    #print(model)
    return get_flops(model)

def main():
    for i in range(4):
        #cand = (0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3)
        print(i, get_cand_flops((i,)*20, (4, )*20))


if __name__ == '__main__':
    main()









