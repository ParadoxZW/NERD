cd nerd
python test.py --gpu='0' \
 --seed=11 --batch_size=36 \
 --ckpt_path='results/05-20-14-21/ckpts/last.pkl' \
 --lr=2.9e-5 --lrr=0.6 --lr_decay=0.3 \
 --b1=0.9 --b2=0.99 \
 --epochs=5 --decay_epoch=3 --threshold=0.5
cd ..
