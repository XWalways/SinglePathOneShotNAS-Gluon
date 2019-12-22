import os
import copy
import sys
import math
import random
import time
import logging
import argparse
import heapq
from mxnet.gluon.data.vision import transforms
from gluoncv.data import imagenet
import mxnet
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import nd
from flops_params import get_cand_flops_params
from network import ShuffleNetV2_OneShot, get_channel_mask
from blocks import BatchNormNAS
import random
import pickle
import numpy as np
sys.setrecursionlimit(10000)
os.environ['MXNET_SAFE_ACCUMULATION'] = '1'
#os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'

stage_repeats = [4, 8, 4, 4]
stage_out_channels = [64, 160, 320, 640]
candidate_scales = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]


parser = argparse.ArgumentParser(description='Searching')
parser.add_argument('--log-dir', type=str, default='search.log')
parser.add_argument('--dtype', type=str, default='float16')
parser.add_argument('--max-epochs', type=int, default=40)
parser.add_argument('--select-num', type=int, default=10)
parser.add_argument('--population-num', type=int, default=50)
parser.add_argument('--m_prob', type=float, default=0.1)
parser.add_argument('--crossover-num', type=int, default=25)
parser.add_argument('--mutation-num', type=int, default=25)
parser.add_argument('--flops-limit', type=float, default=330 * 1e6)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--random-seed', type=int, default=2)
parser.add_argument('--resume-params', type=str, default='supernet_params/0.7459-supernet_imagenet-117-best.params')
parser.add_argument('--checkpoint-name', type=str, default='search_info.pkl')
parser.add_argument('--gpus', type=str, default='0,1,2,3,4,5,6,7')
parser.add_argument('--num-workers', dest='num_workers', type=int, default=60)
parser.add_argument('--data-dir', type=str, default='./data/imagenet')
parser.add_argument('--rec-train', type=str, default='./data/rec/train.rec')
parser.add_argument('--rec-train-idx', type=str, default='./data/rec/train.idx')
parser.add_argument('--rec-val', type=str, default='./data/rec/val.rec')
parser.add_argument('--rec-val-idx', type=str, default='./data/rec/val.idx')
parser.add_argument('--use-rec', action='store_true')
parser.add_argument('--log-interval', type=int, default=50)
parser.add_argument('--input-size', type=int, default=224)
parser.add_argument('--crop-ratio', type=float, default=0.875)
args = parser.parse_args()

filehandler = logging.FileHandler(args.log_dir)
streamhandler = logging.StreamHandler()
logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logger.addHandler(filehandler)
logger.addHandler(streamhandler)
logger.info(args)

choice = lambda x: x[np.random.randint(len(x))] if isinstance(
    x, tuple) else choice(tuple(x))


def get_data_rec(rec_train, rec_train_idx, rec_val, rec_val_idx, batch_size, num_workers, seed):
    rec_train = os.path.expanduser(rec_train)
    rec_train_idx = os.path.expanduser(rec_train_idx)
    rec_val = os.path.expanduser(rec_val)
    rec_val_idx = os.path.expanduser(rec_val_idx)
    jitter_param = 0.4
    lighting_param = 0.1
    input_size = args.input_size
    crop_ratio = args.crop_ratio if args.crop_ratio > 0 else 0.875
    resize = int(math.ceil(input_size / crop_ratio))
    mean_rgb = [123.68, 116.779, 103.939]
    std_rgb = [58.393, 57.12, 57.375]

    def batch_fn(batch, ctx):
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        return data, label

    train_data = mx.io.ImageRecordIter(
        path_imgrec=rec_train,
        path_imgidx=rec_train_idx,
        preprocess_threads=num_workers,
        shuffle=True,
        batch_size=batch_size,

        data_shape=(3, input_size, input_size),
        mean_r=mean_rgb[0],
        mean_g=mean_rgb[1],
        mean_b=mean_rgb[2],
        std_r=std_rgb[0],
        std_g=std_rgb[1],
        std_b=std_rgb[2],
        rand_mirror=True,
        random_resized_crop=True,
        max_aspect_ratio=4. / 3.,
        min_aspect_ratio=3. / 4.,
        max_random_area=1,
        min_random_area=0.08,
        brightness=jitter_param,
        saturation=jitter_param,
        contrast=jitter_param,
        pca_noise=lighting_param,
        seed=seed,
        seed_aug=seed,
        shuffle_chunk_seed=seed,

    )
    val_data = mx.io.ImageRecordIter(
        path_imgrec=rec_val,
        path_imgidx=rec_val_idx,
        preprocess_threads=num_workers,
        shuffle=False,
        batch_size=batch_size,

        resize=resize,
        data_shape=(3, input_size, input_size),
        mean_r=mean_rgb[0],
        mean_g=mean_rgb[1],
        mean_b=mean_rgb[2],
        std_r=std_rgb[0],
        std_g=std_rgb[1],
        std_b=std_rgb[2],
    )
    return train_data, val_data, batch_fn


def get_data_loader(data_dir, batch_size, num_workers):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    jitter_param = 0.4
    lighting_param = 0.1
    input_size = args.input_size
    crop_ratio = args.crop_ratio if args.crop_ratio > 0 else 0.875
    resize = int(math.ceil(input_size / crop_ratio))

    def batch_fn(batch, ctx):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        return data, label

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomFlipLeftRight(),
        transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                                     saturation=jitter_param),
        transforms.RandomLighting(lighting_param),
        transforms.ToTensor(),
        normalize
    ])
    transform_test = transforms.Compose([
        transforms.Resize(resize, keep_ratio=True),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize
    ])

    train_data = gluon.data.DataLoader(
        imagenet.classification.ImageNet(data_dir, train=True).transform_first(transform_train),
        batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)
    val_data = gluon.data.DataLoader(
        imagenet.classification.ImageNet(data_dir, train=False).transform_first(transform_test),
        batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_data, val_data, batch_fn


class EvolutionSearcher(object):
    def __init__(self, args):
        self.args = args
        self.context = [mx.gpu(int(gpu)) for gpu in args.gpus.split(',')] if len(args.gpus.split(',')) > 0 else [mx.cpu()]
        for ctx in self.context:
            mx.random.seed(self.args.random_seed, ctx=ctx)
        np.random.seed(self.args.random_seed)
        random.seed(self.args.random_seed)

        num_gpus = len(self.args.gpus.split(','))
        batch_size = max(1, num_gpus) * self.args.batch_size
        if self.args.use_rec:
            self.train_data, self.val_data, self.batch_fn = get_data_rec(self.args.rec_train, self.args.rec_train_idx,
                                                          self.args.rec_val, self.args.rec_val_idx,
                                                          batch_size, self.args.num_workers, self.args.random_seed)
        else:
            self.train_data, self.val_data, self.batch_fn = get_data_loader(self.args.data_dir, batch_size, self.args.num_workers)

        self.model = ShuffleNetV2_OneShot(search=True)
        self.model.collect_params().load(self.args.resume_params, ctx=self.context, cast_dtype=True, dtype_source='saved')

        self.memory = []
        self.vis_dict = {}
        self.keep_top_k = {self.args.select_num: [], 50: []}
        self.epoch = 0
        self.candidates = []

        self.nr_layer = 20
        self.nr_state = 4
        self.channel_state = 10# len(candidate_scales)

    def save_checkpoint(self):
        if not os.path.exists(self.args.log_dir):
            os.makedirs(self.args.log_dir)
        info = {}
        info['memory'] = self.memory
        info['candidates'] = self.candidates
        info['vis_dict'] = self.vis_dict
        info['keep_top_k'] = self.keep_top_k
        info['epoch'] = self.epoch
        with open(self.args.checkpoint_name, 'wb') as f:
            pickle.dump(info, f)
        print('save checkpoint to', self.args.checkpoint_name)

    def load_checkpoint(self):
        if not os.path.exists(self.args.checkpoint_name):
            return False
        f = open(self.args.checkpoint_name)
        info = pickle.load(f)
        f.close()
        self.memory = info['memory']
        self.candidates = info['candidates']
        self.vis_dict = info['vis_dict']
        self.keep_top_k = info['keep_top_k']
        self.epoch = info['epoch']

        print('load checkpoint from', self.args.checkpoint_name)
        return True


    def is_legal(self, cand):
        assert isinstance(cand, tuple) and len(cand[0]) == self.nr_layer and len(cand[1]) == self.nr_layer
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            return False

        if 'flops' not in info:
            info['flops'], info['params'] = get_cand_flops_params(cand[0], cand[1])

        print(cand, info['flops'])

        if info['flops'] > self.args.flops_limit:
            print('flops limit exceed...')
            return False

        info['err'] = self.get_cand_err(cand)
        info['visited'] = True

        return True

    def update_top_k(self, candidates, k, key, reverse=False):
        assert k in self.keep_top_k
        logger.info('selecting ......')
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]

    def stack_random_cand(self, random_func, batchsize=10):
        while True:
            cands = [random_func() for _ in range(batchsize)]
            print(cands)
            for cand in cands:
                if cand not in self.vis_dict:
                    self.vis_dict[cand] = {}
                info = self.vis_dict[cand]
            for cand in cands:
                yield cand

    def get_random(self):
        logger.info('random selecting ........')
        num = self.args.population_num
        cand_iter = self.stack_random_cand(
            lambda: (tuple(np.random.randint(self.nr_state) for i in range(self.nr_layer)), (9,)*20)) #tuple(np.random.randint(self.channel_state) for i in range(self.nr_layer))
        while len(self.candidates) < num:
            cand = next(cand_iter)
            #print(cand)
            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)
            logger.info('random {}/{}'.format(len(self.candidates), num))
        logger.info('random_num = {}'.format(len(self.candidates)))

    def get_mutation(self):
        k = self.args.select_num
        mutation_num = self.args.mutation_num
        m_prob = self.args.m_prob
        assert k in self.keep_top_k
        logger.info('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = list(choice(self.keep_top_k[k]))
            for i in range(len(cand)):
                cand[0] = list(cand[0])
                cand[1] = list(cand[1])
            for i in range(self.nr_layer):
                if np.random.random_sample() < m_prob:
                    cand[0][i] = np.random.randint(self.nr_state)
                    cand[1][i] = 4 #np.random.randint(self.channel_state) # if you want to search number of channels
            return (tuple(cand[0]), tuple(cand[1]))

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            logger.info('mutation {}/{}'.format(len(res), mutation_num))

        logger.info('mutation_num = {}'.format(len(res)))
        return res

    def get_crossover(self):
        k = self.args.select_num
        crossover_num = self.args.crossover_num
        assert k in self.keep_top_k
        logger.info('crossover ......')
        res = []
        iter = 0
        max_iters = 10 * crossover_num

        def random_func():
            p1 = choice(self.keep_top_k[k])
            p2 = choice(self.keep_top_k[k])
            return (tuple(choice([i, j]) for i, j in zip(p1[0], p2[0])), tuple(choice([i, j]) for i, j in zip(p1[1], p2[1])))

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < crossover_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            logger.info('crossover {}/{}'.format(len(res), crossover_num))

        logger.info('crossover_num = {}'.format(len(res)))
        return res

    def get_cand_err(self, cand, update_images=20000):
        architecture = mxnet.nd.array(cand[0]).astype(dtype=self.args.dtype, copy=False)
        channel_mask = get_channel_mask(cand[1], stage_repeats, stage_out_channels, candidate_scales,
                                        dtype=self.args.dtype)
        # Update BN
        if self.args.use_rec:
            self.train_data.reset()
            self.val_data.reset()
            self.model.cast('float32')
        for k,v in self.model._children.items():
            if isinstance(v, BatchNormNAS):
                v.inference_update_stat = True
        for i,batch in enumerate(self.train_data):
            if (i+1) * self.args.batch_size * len(self.context) >= update_images:
                break
            data, _ = self.batch_fn(batch, self.context)
            _ = [self.model(X.astype('float32', copy=False), architecture.as_in_context(X.context).astype('float32',copy=False),
                            channel_mask.as_in_context(X.context).astype('float32',copy=False)) for X in data]
        for k,v in self.model._children.items():
            if isinstance(v, BatchNormNAS):
                v.inference_update_stat = False
        self.model.cast(self.args.dtype)


        logger.info('starting subnet test....')
        acc_top1 = mx.metric.Accuracy()
        acc_top5 = mx.metric.TopKAccuracy(5)
        acc_top1.reset()
        acc_top5.reset()
        for i, batch in enumerate(self.val_data):
            data, label = self.batch_fn(batch, self.context)
            outputs = [self.model(X.astype(self.args.dtype, copy=False), architecture.as_in_context(X.context).astype(self.args.dtype,copy=False),
                                  channel_mask.as_in_context(X.context).astype(self.args.dtype,copy=False)) for X in data]
            acc_top1.update(label, outputs)
            acc_top5.update(label, outputs)
        _, top1 = acc_top1.get()
        _, top5 = acc_top5.get()
        top1_err = 1 - top1
        top5_err = 1 - top5
        logger.info('top1_err: {:.4f} top5_err: {:.4f}'.format(top1_err, top5_err))
        return top1_err, top5_err

    def search(self):
        logger.info('population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
                self.args.population_num, self.args.select_num, self.args.mutation_num, self.args.crossover_num,
                self.args.population_num - self.args.mutation_num - self.args.crossover_num, self.args.max_epochs))
        self.load_checkpoint()
        self.get_random()

        while self.epoch < self.args.max_epochs:
            logger.info('epoch = {}'.format(self.epoch))

            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)

            self.update_top_k(
                self.candidates, k=self.args.select_num, key=lambda x: self.vis_dict[x]['err'])
            self.update_top_k(
                self.candidates, k=50, key=lambda x: self.vis_dict[x]['err'])

            logger.info('epoch = {} : top {} result'.format(
                self.epoch, len(self.keep_top_k[50])))
            for i, cand in enumerate(self.keep_top_k[50]):
                logger.info('No.{} {} Top-1 err = {}'.format(
                    i + 1, cand, self.vis_dict[cand]['err']))
                ops = [i for i in cand]
                logger.info(ops)

            mutation = self.get_mutation()
            crossover = self.get_crossover()

            self.candidates = mutation + crossover

            self.get_random()
            self.epoch += 1
            if self.epoch % 5 == 0:
                self.save_checkpoint()


def main():
    t = time.time()
    searcher = EvolutionSearcher(args)
    searcher.search()
    logger.info('total searching time = {:.2f} hours'.format((time.time() - t) / 3600))

if __name__ == '__main__':
    main()
