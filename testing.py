import mxnet
def test_supernet():
    """
    Test supernet(network.py)
    """
    from network import ShuffleNetV2_OneShot, get_channel_mask
    stage_repeats = [4, 8, 4, 4]
    stage_out_channels = [64, 160, 320, 640]
    candidate_scales = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    architecture = [0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2]
    architecture = mxnet.nd.array(architecture).astype(dtype='float32', copy=False)
    channel_choice = (4,)*20
    channel_mask = get_channel_mask(channel_choice, stage_repeats, stage_out_channels, candidate_scales, dtype='float32')
    model = ShuffleNetV2_OneShot()
    print(model)
    model.hybridize()
    model._initialize(ctx=mxnet.cpu())
    test_data = mxnet.nd.random.uniform(-1, 1, shape=(5, 3, 224, 224))
    test_outputs = model(test_data, architecture, channel_mask)
    print(test_outputs.shape)
    model.collect_params().save('supernet.params')
    #model.export('supernet')

def test_subnet():
    """
    Test subnet(subnet.py)
    """
    from subnet import ShuffleNetV2_OneShot
    block_choice =(0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2)
    channel_choice = (6, 5, 3, 5, 2, 6, 3, 4, 2, 5, 7, 5, 4, 6, 7, 4, 4, 5, 4, 3)
    model = ShuffleNetV2_OneShot(input_size=224, n_class=1000, architecture=block_choice, channels_idx=channel_choice, act_type='relu', search=False)# define a specific subnet
    model.hybridize()
    model._initialize(ctx=mxnet.cpu())
    print(model)
    test_data = mxnet.nd.random.uniform(-1, 1, shape=(5, 3, 224, 224))
    test_outputs = model(test_data)
    print(test_outputs.shape)
    #model.export('subnet')
    #model.collect_params().save('subnet.params')


def test_flops_params(advanced=False):
    """
    Test the computing of flops and params(flops_params.py/get_flops_params_advanced.py)
    """
    if advanced:
        from get_flops_params_advanced import get_can_flops_params
        for i in range(4):
            #cand = (2, 1, 0, 1, 2, 0, 2, 0, 2, 0, 2, 3, 0, 0, 0, 0, 3, 2, 3, 3)
            print(i, get_cand_flops_params((i,)* 20, (4,) * 20))
    else:
        from flops_params import get_cand_flops_params
        for i in range(4):
            #cand = (2, 1, 0, 1, 2, 0, 2, 0, 2, 0, 2, 3, 0, 0, 0, 0, 3, 2, 3, 3)
            print(i, get_cand_flops_params((i,)* 20, (4,) * 20))


def test_dali():
    """
    Testing dali(dali.py)
    """
    from dali import get_data_rec
    image_shape = (3, 224, 224)
    crop_ratio = 0.875
    # Check Your Dataset Path
    train_rec_file = 'data/rec/train.rec'
    train_rec_file_idx = 'data/rec/train.idx'
    val_rec_file = 'data/rec/val.rec'
    val_rec_file_idx = 'data/rec/val.idx'
    batch_size = 2
    num_workers = 2

    train_data = get_data_rec(image_shape, crop_ratio, train_rec_file, train_rec_file_idx,
                  batch_size, num_workers, train=True, shuffle=True,
                  backend='dali-gpu', gpu_ids=[0,1], kv_store='nccl', dtype='float16',
                  input_layout='NCHW')
    val_data = get_data_rec(image_shape, crop_ratio, val_rec_file, val_rec_file_idx,
                            batch_size, num_workers, train=False, shuffle=False,
                            backend='dali-gpu', gpu_ids=[0, 1], kv_store='nccl', dtype='float16',
                            input_layout='NCHW')
    print(train_data)
    print(val_data)


def test_load_supernet_params():
    """
    Testing the load of supernet's params

    """
    from network import ShuffleNetV2_OneShot
    import mxnet
    model = ShuffleNetV2_OneShot(search=True)
    model.collect_params().load('supernet.params', ctx=mxnet.cpu(), cast_dtype=True, dtype_source='saved')
    print("Done!")

# Please Run the Testing Function in Order, Means You need to Comment Out yhe Other
#test_supernet()
#test_subnet()
#test_flops_params()
#test_dali()
#test_load_supernet_params()

'''
import mxnet
from mxnet import ndarray as F
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

