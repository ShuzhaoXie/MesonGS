MAINDIR=/your/path/to/mesongs
DATADIR=/your/path/to/data
CKPT= # use your path
SCENENAME=mic


CUDA_VISIBLE_DEVICES=0 python render.py -s $DATADIR/nerf_synthetic/$SCENENAME \
    --given_ply_path $MAINDIR/output/$CKPT/point_cloud/iteration_0/pc_npz/bins.zip \
    --eval \
    --skip_train \
    -w \
    --dec_npz \
    --scene_name $SCENENAME \
    --csv_path $MAINDIR/exp_data/csv/test_$CKPT.csv \
    --model_path $MAINDIR/output/$CKPT


# --given_ply_path $MAINDIR/output/$CKPT/point_cloud/iteration_0/pc_npz
