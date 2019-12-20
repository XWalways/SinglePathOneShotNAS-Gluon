from mxnet.gluon import nn
from mxnet.gluon.nn import Block
from mxnet.gluon.block import HybridBlock
from mxnet import nd
import random
import mxnet
import numpy as np
from mxnet import ndarray as F

class BatchNormNAS(HybridBlock):
    def __init__(self, axis=1, momentum=0.9, epsilon=1e-5, center=True, scale=True,
                 use_global_stats=False, beta_initializer='zeros', gamma_initializer='ones',
                 running_mean_initializer='zeros', running_variance_initializer='ones',
                 in_channels=0, inference_update_stat=False, **kwargs):
        super(BatchNormNAS, self).__init__(**kwargs)
        self._kwargs = {'axis': axis, 'eps': epsilon, 'momentum': momentum,
                        'fix_gamma': not scale, 'use_global_stats': use_global_stats}
        self.inference_update_stat = inference_update_stat
        if in_channels != 0:
            self.in_channels = in_channels

        self.gamma = self.params.get('gamma', grad_req='write' if scale else 'null',
                                     shape=(in_channels,), init=gamma_initializer,
                                     allow_deferred_init=True,
                                     differentiable=scale)
        self.beta = self.params.get('beta', grad_req='write' if center else 'null',
                                    shape=(in_channels,), init=beta_initializer,
                                    allow_deferred_init=True,
                                    differentiable=center)
        self.running_mean = self.params.get('running_mean', grad_req='null',
                                            shape=(in_channels,),
                                            init=running_mean_initializer,
                                            allow_deferred_init=True,
                                            differentiable=False)
        self.running_var = self.params.get('running_var', grad_req='null',
                                           shape=(in_channels,),
                                           init=running_variance_initializer,
                                           allow_deferred_init=True,
                                           differentiable=False)
        self.momentum = nd.array([self._kwargs['momentum']])
        self.momentum_rest = nd.array([1 - self._kwargs['momentum']])
    def cast(self, dtype):
        if np.dtype(dtype).name == 'float16':
            dtype = 'float32'
        super(BatchNormNAS, self).cast(dtype)

    def hybrid_forward(self, F, x, gamma, beta, running_mean, running_var):
        if self.inference_update_stat:
            mean = x.mean(axis=(0, 2, 3))
            mean_expanded = F.expand_dims(F.expand_dims(F.expand_dims(mean, axis=0), axis=2), axis=3)
            var = F.square(F.broadcast_minus(x, mean_expanded)).mean(axis=(0, 2, 3))

            running_mean = F.add(F.multiply(self.running_mean.data(), self.momentum.as_in_context(x.context)),
                                 F.multiply(mean, self.momentum_rest.as_in_context(x.context)))
            running_var = F.add(F.multiply(self.running_var.data(), self.momentum.as_in_context(x.context)),
                                F.multiply(var, self.momentum_rest.as_in_context(x.context)))
            self.running_mean.set_data(running_mean)
            self.running_var.set_data(running_var)
            return F.BatchNorm(x, gamma, beta, mean, var, name='fwd', **self._kwargs)
        else:
            return F.BatchNorm(x, gamma, beta, running_mean, running_var, name='fwd', **self._kwargs)

class SE(HybridBlock):
    def __init__(self, num_in):
        super(SE, self).__init__()
        with self.name_scope():
            self.channel_attention = nn.HybridSequential(prefix='')
            self.channel_attention.add(nn.GlobalAvgPool2D(),
                                       nn.Dense(num_in // 16, in_units=num_in, use_bias=False),
                                       nn.Activation('relu'),
                                       nn.Dense(num_in, in_units=num_in//16, use_bias=False),
                                       nn.Activation('sigmoid'))
    def hybrid_forward(self, F, x):
        out = self.channel_attention(x)
        return F.broadcast_mul(x, out.expand_dims(axis=2).expand_dims(axis=2))

class Activation(HybridBlock):
    """Activation function used in MobileNetV3"""
    def __init__(self, act_func, **kwargs):
        super(Activation, self).__init__(**kwargs)
        if act_func == "relu":
            self.act = nn.Activation('relu')
        elif act_func == "relu6":
            self.act = ReLU6()
        elif act_func == "hard_sigmoid":
            self.act = HardSigmoid()
        elif act_func == "swish":
            self.act = nn.Swish()
        elif act_func == "hard_swish":
            self.act = HardSwish()
        elif act_func == "leaky":
            self.act = nn.LeakyReLU(alpha=0.375)
        else:
            raise NotImplementedError

    def hybrid_forward(self, F, x):
        return self.act(x)


class ReLU6(HybridBlock):
    def __init__(self, **kwargs):
        super(ReLU6, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return F.clip(x, 0, 6, name="relu6")


class HardSigmoid(HybridBlock):
    def __init__(self, **kwargs):
        super(HardSigmoid, self).__init__(**kwargs)
        self.act = ReLU6()

    def hybrid_forward(self, F, x):
        return F.clip(x + 3, 0, 6, name="hard_sigmoid") / 6.


class HardSwish(HybridBlock):
    def __init__(self, **kwargs):
        super(HardSwish, self).__init__(**kwargs)
        self.act = HardSigmoid()

    def hybrid_forward(self, F, x):
        return x * (F.clip(x + 3, 0, 6, name="hard_swish") / 6.)



class NasBlockHybridSequential(nn.HybridSequential):
    """
    Handle the ChannelSelector block, takes block_channel_mask as input
    """
    def __init__(self, prefix=None, params=None):
        super(NasBlockHybridSequential, self).__init__(prefix=prefix, params=params)

    def hybrid_forward(self, F, x, block_channel_mask, *args, **kwargs):
        for block in self._children.values():
            if isinstance(block, ChannelSelector):
                x = block(x, block_channel_mask)
            else:
                x = block(x)
        return x


class ShuffleNasBlock(HybridBlock):
    """
    Overwrite the single shufflenet block

    The architecture(block choice) searchable shufflenet block, takes the block_choice and block_channel_mask as input

    if training supernet or searching subnets, search=True

    if retrining subnet, search=False
    """
    def __init__(self, inp, outp, mid_channels, stride, search=True, **kwargs):
        super(ShuffleNasBlock, self).__init__()
        assert stride in [1, 2]
        self.search = search
        with self.name_scope():
            self.block_sn_3x3 = Shufflenet(inp, outp, mid_channels=mid_channels, ksize=3, stride=stride,
                                           act_type='relu', BatchNorm=BatchNormNAS, search=self.search)
            self.block_sn_5x5 = Shufflenet(inp, outp, mid_channels=mid_channels, ksize=5, stride=stride,
                                           act_type='relu', BatchNorm=BatchNormNAS, search=self.search)
            self.block_sn_7x7 = Shufflenet(inp, outp, mid_channels=mid_channels, ksize=7, stride=stride,
                                           act_type='relu', BatchNorm=BatchNormNAS, search=self.search)
            self.block_sx_3x3 = Shuffle_Xception(inp, outp, mid_channels=mid_channels, stride=stride,
                                                 act_type='relu', BatchNorm=BatchNormNAS, search=self.search)

    def hybrid_forward(self, F, x, block_choice, block_channel_mask, *args, **kwargs):
        if block_choice.__eq__(0).__bool__:
            x = self.block_sn_3x3(x, block_channel_mask)
        elif block_choice.__eq__(1).__bool__ :
            x = self.block_sn_5x5(x, block_channel_mask)
        elif block_choice.__eq__(2).__bool__:
            x = self.block_sn_7x7(x, block_channel_mask)
        elif block_choice.__eq__(3).__bool__:
            x = self.block_sx_3x3(x, block_channel_mask)
        return x

class NasHybridSequential(nn.HybridSequential):
    """
    Overwrite the whole shufflenet body

    The channels and architecture both searchable block, takes the full architecture and full channel_mask as input
    """
    def __init__(self, prefix=None, params=None):
        super(NasHybridSequential, self).__init__(prefix=prefix, params=params)

    def hybrid_forward(self, F, x, architecture, channel_mask):
        nas_index = 0
        for block in self._children.values():
            block_choice = F.slice(architecture, begin=nas_index, end=nas_index + 1)
            block_channel_mask = F.slice(channel_mask, begin=(nas_index, None), end=(nas_index + 1, None))
            x = block(x, block_choice, block_channel_mask)
            nas_index += 1
        return x

class ChannelSelector(HybridBlock):
    """
    Random channel selection throuth channel_mask
    """
    def __init__(self, channel_number):
        super(ChannelSelector, self).__init__()
        self.channel_number = channel_number

    def hybrid_forward(self, F, x, block_channel_mask, *args, **kwargs):
        block_channel_mask = F.slice(block_channel_mask, begin=(None,None), end=(None, self.channel_number))
        block_channel_mask = F.reshape(block_channel_mask, shape=(1, self.channel_number, 1, 1))
        x = F.broadcast_mul(x, block_channel_mask)
        return x

class Shufflenet(HybridBlock):
    def __init__(self, inp, oup, mid_channels, ksize, stride, act_type='relu', BatchNorm=nn.BatchNorm, search=True):
        super(Shufflenet, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        assert ksize in [3, 5, 7]


        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.project_channel = inp // 2 if stride == 1 else inp
        self.main_input_channel = inp // 2 if stride == 1 else inp
        self.main_output_channel = oup - self.project_channel
        self.search = search


        #pw
        with self.name_scope():
            if stride == 1:
                self.channel_shuffle = ShuffleChannels(mid_channel=inp // 2, groups=2)
            if self.search:
                self.branch_main = NasBlockHybridSequential(prefix='branch_main_')
            else:
                self.branch_main = nn.HybridSequential(prefix='branch_main_')
            self.branch_main.add(nn.Conv2D(mid_channels, in_channels=self.main_input_channel, kernel_size=1, strides=1,
                                  padding=0, use_bias=False))
            if self.search:
                self.branch_main.add(ChannelSelector(channel_number=mid_channels))
            self.branch_main.add(BatchNorm(in_channels=mid_channels, momentum=0.1))
            self.branch_main.add(Activation(act_type))

            #dw
            self.branch_main.add(nn.Conv2D(mid_channels, in_channels=mid_channels, kernel_size=ksize, strides=stride, groups=mid_channels,
                                           padding=pad, use_bias=False))
            self.branch_main.add(BatchNorm(in_channels=mid_channels, momentum=0.1))

            #pw_linear
            self.branch_main.add(nn.Conv2D(self.main_output_channel, in_channels=mid_channels, kernel_size=1, strides=1, padding=0, use_bias=False))
            self.branch_main.add(BatchNorm(in_channels=self.main_output_channel, momentum=0.1))
            self.branch_main.add(Activation(act_type))

            #se
            self.branch_main.add(SE(self.main_output_channel))

            if stride == 2:
                self.branch_proj = nn.HybridSequential(prefix='branch_proj_')
                #dw
                self.branch_proj.add(nn.Conv2D(self.project_channel, in_channels=self.project_channel, kernel_size=ksize, strides=stride, groups=self.project_channel,
                                           padding=pad, use_bias=False))
                self.branch_proj.add(BatchNorm(in_channels=self.project_channel, momentum=0.1))

                #pw-linear
                self.branch_proj.add(nn.Conv2D(self.project_channel, in_channels=self.project_channel, kernel_size=1, strides=1, padding=0, use_bias=False))
                self.branch_proj.add(BatchNorm(in_channels=self.project_channel, momentum=0.1))
                self.branch_proj.add(Activation(act_type))

    def hybrid_forward(self, F, old_x, block_channel_mask=None, *args, **kwargs):
        if self.search:
            if self.stride == 1:
                x_proj, x = self.channel_shuffle(old_x)

                return F.concat(x_proj, self.branch_main(x, block_channel_mask), dim=1)
            elif self.stride == 2:
                x_proj = old_x
                x = old_x
                return F.concat(self.branch_proj(x_proj), self.branch_main(x, block_channel_mask), dim=1)
        else:
            if self.stride == 1:
                x_proj, x = self.channel_shuffle(old_x)

                return F.concat(x_proj, self.branch_main(x), dim=1)
            elif self.stride == 2:
                x_proj = old_x
                x = old_x
                return F.concat(self.branch_proj(x_proj), self.branch_main(x), dim=1)

class Shuffle_Xception(HybridBlock):
    def __init__(self, inp, oup, mid_channels, stride, act_type='relu', BatchNorm=nn.BatchNorm, search=True):
        super(Shuffle_Xception, self).__init__()

        assert stride in [1, 2]


        self.stride = stride
        self.ksize = 3
        self.pad = 1
        self.project_channel = inp // 2 if stride == 1 else inp
        self.main_input_channel = inp // 2 if stride == 1 else inp
        self.main_output_channel = oup - self.project_channel
        self.search = search

        with self.name_scope():
            if stride == 1:
                self.channel_shuffle = ShuffleChannels(mid_channel=inp // 2, groups=2)
            if self.search:
                self.branch_main = NasBlockHybridSequential(prefix='branch_main_')
            else:
                self.branch_main = nn.HybridSequential(prefix='branch_main_')
            #dw
            self.branch_main.add(nn.Conv2D(self.main_input_channel, in_channels=self.main_input_channel, kernel_size=3, strides=stride, padding=1, groups=self.main_input_channel, use_bias=False))
            self.branch_main.add(BatchNorm(in_channels=self.main_input_channel, momentum=0.1))
            #pw
            self.branch_main.add(nn.Conv2D(mid_channels, in_channels=self.main_input_channel, kernel_size=1, strides=1, padding=0, use_bias=False))
            if self.search:
                self.branch_main.add(ChannelSelector(channel_number=mid_channels))
            self.branch_main.add(BatchNorm(in_channels=mid_channels, momentum=0.1))
            self.branch_main.add(Activation(act_type))

            #dw
            self.branch_main.add(nn.Conv2D(mid_channels, in_channels=mid_channels, kernel_size=3, strides=1, padding=1, groups=mid_channels, use_bias=False))
            self.branch_main.add(BatchNorm(in_channels=mid_channels, momentum=0.1))

            #pw
            self.branch_main.add(nn.Conv2D(mid_channels, in_channels=mid_channels, kernel_size=1, strides=1, padding=0, use_bias=False))
            self.branch_main.add(BatchNorm(in_channels=mid_channels, momentum=0.1))
            self.branch_main.add(Activation(act_type))

            #dw
            self.branch_main.add(nn.Conv2D(mid_channels, in_channels=mid_channels, kernel_size=3, strides=1, padding=1, groups=mid_channels, use_bias=False))
            self.branch_main.add(BatchNorm(in_channels=mid_channels, momentum=0.1))

            #pw
            self.branch_main.add(nn.Conv2D(self.main_output_channel, in_channels=mid_channels, kernel_size=1, strides=1, padding=0, use_bias=False))
            self.branch_main.add(BatchNorm(in_channels=self.main_output_channel, momentum=0.1))
            self.branch_main.add(Activation(act_type))

            #se
            self.branch_main.add(SE(self.main_output_channel))

            if stride == 2:
                self.branch_proj = nn.HybridSequential(prefix='branch_proj_')
                #dw
                self.branch_proj.add(nn.Conv2D(self.project_channel, in_channels=self.project_channel, kernel_size=3, strides=stride, padding=1, groups=self.project_channel, use_bias=False))
                self.branch_proj.add(BatchNorm(in_channels=self.project_channel, momentum=0.1))

                #pw_linear
                self.branch_proj.add(nn.Conv2D(self.project_channel, in_channels=self.project_channel, kernel_size=1, strides=1, padding=0, use_bias=False))
                self.branch_proj.add(BatchNorm(in_channels=self.project_channel, momentum=0.1))
                self.branch_proj.add(Activation(act_type))

    def hybrid_forward(self, F, old_x, block_channel_mask=None, *args, **kwargs):
        if self.search:
            if self.stride == 1:
                x_proj, x = self.channel_shuffle(old_x)

                return F.concat(x_proj, self.branch_main(x, block_channel_mask), dim=1)
            elif self.stride == 2:
                x_proj = old_x
                x = old_x
                return F.concat(self.branch_proj(x_proj), self.branch_main(x, block_channel_mask), dim=1)
        else:
            if self.stride == 1:
                x_proj, x = self.channel_shuffle(old_x)

                return F.concat(x_proj, self.branch_main(x), dim=1)
            elif self.stride == 2:
                x_proj = old_x
                x = old_x
                return F.concat(self.branch_proj(x_proj), self.branch_main(x), dim=1)

class ShuffleChannels(HybridBlock):
    def __init__(self, mid_channel, groups=2, **kwargs):
        super(ShuffleChannels, self).__init__()
        # For ShuffleNet v2, groups is always set 2
        assert groups == 2
        self.groups = groups
        self.mid_channel = mid_channel

    def hybrid_forward(self, F, x, *args, **kwargs):
        # batch_size, channels, height, width = x.shape
        # assert channels % 2 == 0
        # mid_channels = channels // 2
        data = F.reshape(x, shape=(0, -4, self.groups, -1, -2))
        data = F.swapaxes(data, 1, 2)
        data = F.reshape(data, shape=(0, -3, -2))
        data_project = F.slice(data, begin=(None, None, None, None), end=(None, self.mid_channel, None, None))
        data_x = F.slice(data, begin=(None, self.mid_channel, None, None), end=(None, None, None, None))
        return data_project, data_x


