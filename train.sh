# Train standard model 
#python main.py --device=cuda:0

# Model Train 1 (Affect of Patch Size) 
python main.py --device=cuda:0 --patch_size=500 

# Model Train 2 (Affect of batch size with old path size)
python main.py --device=cude:0 --batch=16 --patch_size=250 

# Model Train 3 (Affect of learning rate on old patch size)
python main.py --device=cuda:0 --lr=1e-3 --patch_size=250 

# Model Train 4 (Affect of loss on old patch_size)
python main.py --device=cuda:0 --loss=bce --patch_size=250 

# MODEL TRAIN FOR DOING AUGMENTATIONS ON THE FLY, NOT IN THE PREPROCESS.PY FILE (500 patch size)
python main.py --device=cuda:0 --augfly
