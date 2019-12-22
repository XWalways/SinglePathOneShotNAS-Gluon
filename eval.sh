python eval.py \
  --lr 0.4 --lr-mode cosine --num-epochs 120 --batch-size 128 --num-gpus 8 -j 60 \
  --warmup-epochs 5 --use-rec \
  --no-wd --label-smoothing \
  --teacher resnet152_v1d --temperature 1 --hard-weight 0.5
