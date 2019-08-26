CMD="python -m src_ortho.main --encoder_only=True --e_lr 1e-5 --log_img_step 100000 --e_loss_weight 60. --batch_size=32 --use_3d_label True --e_3d_weight 60. --e_pose_weight 60. --e_shape_weight 60. --epoch 20"

echo $CMD
$CMD
