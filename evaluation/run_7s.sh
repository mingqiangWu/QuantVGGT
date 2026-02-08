CUDA_VISIBLE_DEVICES=4 python Quant_VGGT/vggt/evaluation/run_7andN.py\
    --model_path Quant_VGGT/VGGT-1B/model_tracker_fixed_e20.pt \
    --co3d_dir co3d_datasets/ \
    --co3d_anno_dir co3d_v2_annotations/ \
    --dtype quarot_w4a4\
    --each_nsamples 10 \
    --lwc \
    --lac \
    --exp_name quant_w4a4\
    --cache_path Quant_VGGT/vggt/evaluation/outputs/cache_data.pt \
    --class_mode all \
    --output_dir "Quant_VGGT/vggt/eval_results" \
    --kf 100 \
    --dataset nr \


