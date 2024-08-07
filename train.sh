
# python main.py --device=cuda:2 --model=unet --loss=all
# python main.py --inference --model=unet --checkpoint=saved_best_0.0001_8_5_0.01_unet_False_all --device=cuda:2

# # python Data/preprocess.py
python main.py --device=cuda:2 --model=xlstm --loss=all
python main.py --inference --model=xlstm --checkpoint=saved_best_0.0001_8_5_0.01_xlstm_False_all --device=cuda:2

python main.py --device=cuda:2 --model=attxlstm --loss=all
python main.py --inference --model=attxlstm --checkpoint=saved_best_0.0001_8_5_0.01_attxlstm_False_all --device=cuda:2
