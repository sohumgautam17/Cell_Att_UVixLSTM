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


################################
#!/bin/bash

aug_fly=(0 1)

patch_size=(250 512)

batch_sizes=(4 8 16 32)

lr=(1e-3 1e-4 1e-5)



# Loop over each batch size
for aug_fly in "${aug_fly[@]}"
do
    # Loop over each num_embeddings configuration
    for patch_size in "${patch_size[@]}"
    do
        # Loop over the seeds array for each configuration
        for batch_size in "${batch_sizes[@]}"
        do
            for lr in "${lr[@]}"
            do

            echo "Running with augmentatation on the fly=$aug_fly, patch_size=$patch_size, batch_size=$batch_size, learning rate=$lr"
            python main.py --device=cuda:0 

        done
    done
done