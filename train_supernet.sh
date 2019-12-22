python train_supernet.py \
  --mode hybrid \
  --lr 0.4 --lr-mode cosine --num-epochs 120 --batch-size 128 --num-gpus 8 -j 60 \
  --warmup-epochs 5 --dtype float16 \
  --use-rec --no-wd --label-smoothing 

