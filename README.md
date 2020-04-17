# Named Entity Representations for Disambiguation (NERD)
A Pytorch implementation of NERD for training, testing and inference.

## Architecture
![avatar](imgs/nerd.png "NERD")
The offsets of entities are obtained by string match due to the simplicity of the given entity sets.

## Log
run `bash train.sh | tee log.txt`
```
args:
Namespace(b1=0.9, b2=0.99, batch_size=24, decay_epoch=3, epochs=5, gpu='1,0', lr=3e-05, lr_decay=0.3, lrr=0.6, seed=888, threshold=0.5)
Load datasets!
datasets successfully loaded!

config model and optim...
start training!

Epoch: 0, LR: 3.000000e-05
evaluating!
evaluate ../results/04-16-05-58/0.json
R: 0.947853, P: 0.947853, F1, 0.947853


Epoch: 1, LR: 3.000000e-05
evaluating!
evaluate ../results/04-16-05-58/1.json
R: 0.950920, P: 0.950920, F1, 0.950920


Epoch: 2, LR: 3.000000e-05
evaluating!
evaluate ../results/04-16-05-58/2.json
R: 0.950307, P: 0.950307, F1, 0.950307


Epoch: 3, LR: 9.000000e-06
evaluating!
evaluate ../results/04-16-05-58/3.json
R: 0.954601, P: 0.954601, F1, 0.954601


Epoch: 4, LR: 9.000000e-06
evaluating!
evaluate ../results/04-16-05-58/4.json
R: 0.963190, P: 0.963190, F1, 0.963190
```