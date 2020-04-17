cd nerd
python train.py --gpu='1,0' \
 --seed=888 --batch_size=$bs \
 --lr=$lr --lrr=$(echo "$lrr * 0.1"|bc) --lr_decay=$ld \
 --b1=0.9 --b2=$b \
 --epochs=5 --decay_epoch=3
cd ..
