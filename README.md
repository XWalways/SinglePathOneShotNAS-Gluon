# SinglePathOneShotNAS-Gluon

## Gloun Version of SinglePathOneShotNAS

Official [Pytorch](https://github.com/megvii-model/SinglePathOneShot) Implementation 

[Single Path One-Shot Neural Architecture Search with Uniform Sampling](https://arxiv.org/abs/1904.00420)


## Prepare

`pip install mxnet-cu101mkl`(depend on you CUDA version)

`pip install gluoncv`

`pip install mxboard`

`pip install mxop`(Optional)

## SuperNet

Train supernet with:`sh train_supernet.sh`

Remark: Change `data-dir` or `rec-train`& `rec-train-idx` & `rec-val` & `rec-val-idx` in `train_supernet.py` or `train_supernet.sh` before training.


## Search

Search subnet with flops/params limits:`sh search.sh`

Remark: Change `resume-params` in `search.py` or `search.sh`, also you should change dataset dir in `search.py` or `search.sh` before searching, change flops/params limit if you like.

## Evaluate

Retrain the best subnet:`sh eval.sh`

Remark: Change dataset dir in `eval.py` or `eval.sh` before retraining.



## Reference

https://github.com/megvii-model/SinglePathOneShot

https://gluon-cv.mxnet.io/model_zoo/classification.html

https://github.com/CanyonWind/Single-Path-One-Shot-NAS-MXNet

https://github.com/hey-yahei/OpSummary.MXNet




