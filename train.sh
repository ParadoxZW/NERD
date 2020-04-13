cd nerd
python train.py --gpu='1'\
--seed=888 --batch_size=32\
--lr=0.001 --lrr=0.1 --lr_decay=0.1\
--b1=0.9 --b2=0.98\
--epochs=10 --decay_epoch=7 | tee ../log.txt
cd ..