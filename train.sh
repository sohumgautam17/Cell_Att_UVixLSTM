python main.py --device=cuda:2 --dataset=clip_dataset --model=clip_xlstm --loss=all --batch=32 
python main.py --inference --dataset=clip_dataset --model=clip_xlstm --model_checkpoint=saved_best_0.0001_32_5_0.01_clip_xlstm_all --device=cuda:2

# python main.py --device=cuda:2 --model=xlstm --loss=all --batch=32 --lr=1e-3
# python main.py --inference --model=xlstm --checkpoint=saved_best_0.001_32_5_0.01_xlstm_False_all --device=cuda:2

# python main.py --device=cuda:2 --model=hovernet --loss=all --batch=32 --lr=1e-3
# python main.py --inference --model=xlstm --checkpoint=saved_best_0.001_32_5_0.01_hovernet_False_all --device=cuda:2
