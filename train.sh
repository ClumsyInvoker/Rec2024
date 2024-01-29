python main_pretrain.py --dataset MovieLens1m --train_dir DIN --num_epochs 201 --device cuda --l2_emb 0.01
python main_pretrain.py --dataset MovieLens100k --train_dir DIN --num_epochs 201 --device cuda --l2_emb 0.01

python main_pretrain.py --dataset MovieLens100k --train_dir DSSM --num_epochs 201 --device cuda
python main.py --dataset MovieLens100k --train_dir DSSM_PTCR --num_epochs 201 --device cuda --pretrain_model_path ./MovieLens100k_DSSM/epoch=5.lr=0.001.layer=2.head=1.hidden=50.maxlen=50.pth