cd nerd
python train.py --gpu='1,0' \
 --seed=888 --batch_size=$bs \
 --lr=4.5e-5 --lrr=0.2 --lr_decay=0.3 \
 --b1=0.9 --b2=32 \
 --epochs=5 --decay_epoch=3
cd ..
