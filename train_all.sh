cd nerd
python train_all.py --gpu='1' --version='3' \
 --seed=888 --batch_size=24 \
 --lr=2.9e-5 --lrr=0.6 --lr_decay=0.3 \
 --b1=0.9 --b2=0.99 \
 --epochs=5 --decay_epoch=3 --threshold=0.5
cd ..
