cd nerd
python train.py --gpu='1,0' \
 --seed=888 --batch_size=24 \
 --lr=3e-5 --lrr=0.6 --lr_decay=0.3 \
 --b1=0.9 --b2=0.99 \
 --epochs=5 --decay_epoch=3
cd ..
