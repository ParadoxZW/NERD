# Named Entity Representations for Disambiguation (NERD)
A Pytorch implementation of NERD for training, testing and inference.

## Architecture
![avatar](imgs/nerd.png "NERD")
The offsets of entities are obtained by string match due to the simplicity of the given entity sets.

## Quick Start
I provide several bash scripts for different purpose and application scenarios. Just simply run:
```Bash
bash train.sh  # train a model
bash test.sh   # process test.txt and get a result json
bash deploy.sh # start algorithm server
```
You are free to modify the configs or parameters in these scripts for your customized experiments and applications.

## Log
run `bash train.sh | tee log.txt`
```
Calling BertTokenizer.from_pretrained() with the path to a single file or url is deprecated
args:
Namespace(b1=0.9, b2=0.99, batch_size=24, decay_epoch=3, epochs=5, gpu='1,0', lr=3e-05, lr_decay=0.3, lrr=0.6, seed=888, threshold=0.5)
Load datasets!
datasets successfully loaded!

config model and optim...
start training!

Epoch: 0, LR: 3.000000e-05
loss 0.046479: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 655/655 [04:13<00:00,  2.63it/s]
acc 0.966667: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 51/51 [00:07<00:00,  6.74it/s]
evaluating!
evaluate ../results/04-17-10-35/0.json
R: 0.947853, P: 0.947853, F1, 0.947853


Epoch: 1, LR: 3.000000e-05
loss 0.079792: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 655/655 [04:12<00:00,  2.61it/s]
acc 0.966667: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 51/51 [00:07<00:00,  6.88it/s]
evaluating!
evaluate ../results/04-17-10-35/1.json
R: 0.950920, P: 0.950920, F1, 0.950920


Epoch: 2, LR: 3.000000e-05
loss 0.108233: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 655/655 [04:12<00:00,  2.64it/s]
acc 1.000000: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 51/51 [00:07<00:00,  6.82it/s]
evaluating!
evaluate ../results/04-17-10-35/2.json
R: 0.950307, P: 0.950307, F1, 0.950307


Epoch: 3, LR: 9.000000e-06
loss 0.001395: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 655/655 [04:12<00:00,  2.65it/s]
acc 0.966667: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 51/51 [00:07<00:00,  6.78it/s]
evaluating!
evaluate ../results/04-17-10-35/3.json
R: 0.954601, P: 0.954601, F1, 0.954601


Epoch: 4, LR: 9.000000e-06
loss 0.005694: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 655/655 [04:12<00:00,  2.62it/s]
acc 1.000000: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 51/51 [00:07<00:00,  6.90it/s]
evaluating!
evaluate ../results/04-17-10-35/4.json
R: 0.963190, P: 0.963190, F1, 0.963190
```