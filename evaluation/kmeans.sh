CUDA_VISIBLE_DEVICES=6 python Quant_VGGT/vggt/evaluation/make_calibation.py \
    --model_path Quant_VGGT/VGGT-1B/model_tracker_fixed_e20.pt \
    --co3d_dir co3d_datasets/ \
    --co3d_anno_dir/datasets/co3d_v2_annotations/ \
    --seed 0 \
    --each_nsamples 10 \
    --cache_path Quant_VGGT/vggt/evaluation/outputs/all_calib_data.pt \
    --save_path Quant_VGGT/vggt/evaluation/outputs/total_20_calib_data.pt \
    --class_mode all \
    --kmeans_n 5 \
    --kmeans_m 4 \




