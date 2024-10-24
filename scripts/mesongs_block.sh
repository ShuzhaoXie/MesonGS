SCENE=mic
ITERS=8000
DATAPATH=/your/path/to/nerf_synthetic/$SCENE
INITIALPATH=/your/path/to/output/$SCENE/point_cloud/iteration_30000/point_cloud.ply
CONFIG=config3
CSVPATH=/your/path/to/exp_data/csv/$SCENE\_$CONFIG.csv
SAVEPATH=/your/path/to/output/$SCENE\_$CONFIG

LSEG=0 # using the pre-written config, so do not use the LSED config.
CB=0 # same as LSEG
DEPTH=0 # same as LSEG


CUDA_VISIBLE_DEVICES=3 python mesongs.py -s $DATAPATH \
    --given_ply_path $INITIALPATH \
    --num_bits 8 \
    --save_imp --eval \
    --iterations $ITERS \
    --finetune_lr_scale 1 \
    --convert_SHs_python \
    --percent 0 \
    --codebook_size $CB \
    --steps 1000 \
    --scene_imp $SCENE \
    --depth $DEPTH \
    --raht \
    --clamp_color \
    --per_block_quant \
    --lseg $LSEG \
    --use_indexed \
    --debug \
    --hyper_config $CONFIG \
    --csv_path $CSVPATH \
    --model_path $SAVEPATH
