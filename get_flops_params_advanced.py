from flops_params import ShuffleNetV2_OneShot

def get_cand_flops(cand, channels_idx):
    model = ShuffleNetV2_OneShot(input_size=224, n_class=1000, architecture=cand, channels_idx=channels_idx, act_type='relu', search=False)
    #print(model)
    return get_flops(model)

def get_flops(model):
    # use the package mxop(https://github.com/hey-yahei/OpSummary.MXNet)
    # Maybe More Accurate

    from mxop.gluon import count_ops
    from mxop.gluon import count_params
    op_counter = count_ops(model, input_size=(1,3,224,224))['muls']
    count_params = count_params(model)
    return op_counter, count_params

def main():
    for i in range(4):
        #cand = (2, 1, 0, 1, 2, 0, 2, 0, 2, 0, 2, 3, 0, 0, 0, 0, 3, 2, 3, 3)
        print(i, get_cand_flops((i,)*20, (4, )*20))


if __name__ == '__main__':
    main()