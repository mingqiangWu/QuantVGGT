
CUDA_VISIBLE_DEVICES=4 python Quant_VGGT/vggt/evaluation/run_co3d.py \
    --model_path Quant_VGGT/VGGT-1B/model_tracker_fixed_e20.pt \
    --co3d_dir co3d_datasets/ \
    --co3d_anno_dir co3d_v2_annotations/ \
    --dtype quarot_w4a4\
    --seed 0 \
    --lac \
    --lwc \
    --cache_path Quant_VGGT/vggt/evaluation/outputs/cache_data.pt \
    --class_mode all \
    --each_nsamples 1 \
    --exp_name test \
    --resume_qs \




