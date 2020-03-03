python3 draw_raw_detection_map.py \
    --output_dir '/data/KITTI_object_tracking/spatio-temporal-map/draw_pics/raw_detection_ST_Map' \
    --detection_data_pkl '/data/KITTI_object_tracking/Carla_data/label/NetworkEvaluationDataset/carla-episode2-frame600-vehicle18-pedestrian7.pkl' \
    --carla_flag \
    --frame global \
    --map_duration_frame 10 \
    --seq_start 0 \
    --seq_end 0 \
    --save_3d

# python3 draw_raw_detection_map.py \
#     --output_dir '/data/KITTI_object_tracking/spatio-temporal-map/draw_pics/raw_detection_ST_Map' \
#     --detection_data_pkl '/data/KITTI_object_tracking/results_PointRCNNTrackNet/detection_pkl/training_result_RPN035_PCNN085.pkl' \
#     --frame global \
#     --map_duration_frame 10 \
#     --seq_start 4 \
#     --seq_end 4 \
#     --save_3d