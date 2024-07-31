MAINDIR=/your/path/to/mesongs
DATADIR=/your/path/to/data
SCENENAME=mic

CUDA_VISIBLE_DEVICES=0 python mesongs.py -s $DATADIR/nerf_synthetic/$SCENENAME \
    --given_ply_path $MAINDIR/output/$SCENENAME/point_cloud/iteration_30000/point_cloud.ply \
    -w --eval \
    --iteration 10 \
    --scene_name $SCENENAME \
    --skip_post_eval \
    --csv_path $MAINDIR/exp_data/csv/meson_$SCENENAME.csv \
    --model_path $MAINDIR/output/meson_$SCENENAME
