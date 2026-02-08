# coding=utf-8
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
import torch
import numpy as np
import gzip
import json
import random
import logging
import warnings
from vggt.models.vggt import VGGT
from vggt.utils.rotation import mat_to_quat
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import closed_form_inverse_se3
from ba import run_vggt_with_ba
import argparse
import logging
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict
import torch.nn.functional as F
from quarot.utils import quarot_smooth_quant_model, set_ignore_quantize,load_qs_parameters
from quarot.args_utils import get_config
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import time
from compare import compare_models,compare_model_structure,print_model_comparison_summary,verify_models_identical

# Suppress DINO v2 logs
logging.getLogger("dinov2").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="dinov2")

# Set computation precision
torch.set_float32_matmul_precision('highest')
torch.backends.cudnn.allow_tf32 = False

from sklearn.cluster import MiniBatchKMeans,KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def convert_pt3d_RT_to_opencv(Rot, Trans):
    """
    Convert Point3D extrinsic matrices to OpenCV convention.

    Args:
        Rot: 3D rotation matrix in Point3D format
        Trans: 3D translation vector in Point3D format

    Returns:
        extri_opencv: 3x4 extrinsic matrix in OpenCV format
    """
    rot_pt3d = np.array(Rot)
    trans_pt3d = np.array(Trans)

    trans_pt3d[:2] *= -1
    rot_pt3d[:, :2] *= -1
    rot_pt3d = rot_pt3d.transpose(1, 0)
    extri_opencv = np.hstack((rot_pt3d, trans_pt3d[:, None]))
    return extri_opencv


def build_pair_index(N, B=1):
    """
    Build indices for all possible pairs of frames.

    Args:
        N: Number of frames
        B: Batch size

    Returns:
        i1, i2: Indices for all possible pairs
    """
    i1_, i2_ = torch.combinations(torch.arange(N), 2, with_replacement=False).unbind(-1)
    i1, i2 = [(i[None] + torch.arange(B)[:, None] * N).reshape(-1) for i in [i1_, i2_]]
    return i1, i2


def rotation_angle(rot_gt, rot_pred, batch_size=None, eps=1e-15):
    """
    Calculate rotation angle error between ground truth and predicted rotations.

    Args:
        rot_gt: Ground truth rotation matrices
        rot_pred: Predicted rotation matrices
        batch_size: Batch size for reshaping the result
        eps: Small value to avoid numerical issues

    Returns:
        Rotation angle error in degrees
    """
    q_pred = mat_to_quat(rot_pred)
    q_gt = mat_to_quat(rot_gt)

    loss_q = (1 - (q_pred * q_gt).sum(dim=1) ** 2).clamp(min=eps)
    err_q = torch.arccos(1 - 2 * loss_q)

    rel_rangle_deg = err_q * 180 / np.pi

    if batch_size is not None:
        rel_rangle_deg = rel_rangle_deg.reshape(batch_size, -1)

    return rel_rangle_deg


def translation_angle(tvec_gt, tvec_pred, batch_size=None, ambiguity=True):
    """
    Calculate translation angle error between ground truth and predicted translations.

    Args:
        tvec_gt: Ground truth translation vectors
        tvec_pred: Predicted translation vectors
        batch_size: Batch size for reshaping the result
        ambiguity: Whether to handle direction ambiguity

    Returns:
        Translation angle error in degrees
    """
    rel_tangle_deg = compare_translation_by_angle(tvec_gt, tvec_pred)
    rel_tangle_deg = rel_tangle_deg * 180.0 / np.pi

    if ambiguity:
        rel_tangle_deg = torch.min(rel_tangle_deg, (180 - rel_tangle_deg).abs())

    if batch_size is not None:
        rel_tangle_deg = rel_tangle_deg.reshape(batch_size, -1)

    return rel_tangle_deg


def compare_translation_by_angle(t_gt, t, eps=1e-15, default_err=1e6):
    """
    Normalize the translation vectors and compute the angle between them.

    Args:
        t_gt: Ground truth translation vectors
        t: Predicted translation vectors
        eps: Small value to avoid division by zero
        default_err: Default error value for invalid cases

    Returns:
        Angular error between translation vectors in radians
    """
    t_norm = torch.norm(t, dim=1, keepdim=True)
    t = t / (t_norm + eps)

    t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
    t_gt = t_gt / (t_gt_norm + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))

    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t


def calculate_auc_np(r_error, t_error, max_threshold=30):
    """
    Calculate the Area Under the Curve (AUC) for the given error arrays using NumPy.

    Args:
        r_error: numpy array representing R error values (Degree)
        t_error: numpy array representing T error values (Degree)
        max_threshold: Maximum threshold value for binning the histogram

    Returns:
        AUC value and the normalized histogram
    """
    error_matrix = np.concatenate((r_error[:, None], t_error[:, None]), axis=1)
    max_errors = np.max(error_matrix, axis=1)
    bins = np.arange(max_threshold + 1)
    histogram, _ = np.histogram(max_errors, bins=bins)
    num_pairs = float(len(max_errors))
    normalized_histogram = histogram.astype(float) / num_pairs
    return np.mean(np.cumsum(normalized_histogram)), normalized_histogram


def se3_to_relative_pose_error(pred_se3, gt_se3, num_frames):
    """
    Compute rotation and translation errors between predicted and ground truth poses.

    Args:
        pred_se3: Predicted SE(3) transformations
        gt_se3: Ground truth SE(3) transformations
        num_frames: Number of frames

    Returns:
        Rotation and translation angle errors in degrees
    """
    pair_idx_i1, pair_idx_i2 = build_pair_index(num_frames)

    # Compute relative camera poses between pairs
    # We use closed_form_inverse to avoid potential numerical loss by torch.inverse()
    relative_pose_gt = closed_form_inverse_se3(gt_se3[pair_idx_i1]).bmm(
        gt_se3[pair_idx_i2]
    )
    relative_pose_pred = closed_form_inverse_se3(pred_se3[pair_idx_i1]).bmm(
        pred_se3[pair_idx_i2]
    )

    # Compute the difference in rotation and translation
    rel_rangle_deg = rotation_angle(
        relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3]
    )
    rel_tangle_deg = translation_angle(
        relative_pose_gt[:, :3, 3], relative_pose_pred[:, :3, 3]
    )

    return rel_rangle_deg, rel_tangle_deg


def align_to_first_camera(camera_poses):
    """
    Align all camera poses to the first camera's coordinate frame.

    Args:
        camera_poses: Tensor of shape (N, 4, 4) containing camera poses as SE3 transformations

    Returns:
        Tensor of shape (N, 4, 4) containing aligned camera poses
    """
    first_cam_extrinsic_inv = closed_form_inverse_se3(camera_poses[0][None])
    aligned_poses = torch.matmul(camera_poses, first_cam_extrinsic_inv)
    return aligned_poses

# ⭐
def setup_args():
    """Set up command-line arguments for the CO3D evaluation script."""
    parser = argparse.ArgumentParser(description='Test VGGT on CO3D dataset')
    parser.add_argument('--debug_mode', type=str,default='all', help='Enable debug mode (only test on specific category)')
    parser.add_argument('--use_ba', action='store_true', default=False, help='Enable bundle adjustment')
    parser.add_argument('--fast_eval', action='store_true', default=False, help='Only evaluate 10 sequences per category')
    parser.add_argument('--min_num_images', type=int, default=50, help='Minimum number of images for a sequence')
    parser.add_argument('--num_frames', type=int, default=10, help='Number of frames to use for testing')
    parser.add_argument('--co3d_dir', type=str, required=True, help='Path to CO3D dataset')
    parser.add_argument('--co3d_anno_dir', type=str, required=True, help='Path to CO3D annotations')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the VGGT model checkpoint')

    parser.add_argument('--dtype', type=str, default='quarot_w6a6', help='Data type for model inference')
    parser.add_argument('--each_nsamples', type=int, default='2', help='每个类选多少个e')
    parser.add_argument('--not_smooth', action='store_true', help='禁用smooth（默认启用）')
    parser.add_argument('--not_rot', action='store_true', help='禁用rot（默认启用）')
    parser.add_argument('--lwc', action='store_true', help='使用lwc')
    parser.add_argument('--lac', action='store_true', help='使用lac')
    parser.add_argument('--rv', action='store_true', help='使用rv')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name')
    parser.add_argument('--cache_path', type=str, default=None, help='加载或者保存特殊名字的cache，可以直接传入')
    parser.add_argument('--save_path', type=str, default='calib_data_final.pt', help='保存下来的cache')

    parser.add_argument('--compile', action='store_true', default=False, help='Compile the model')
    parser.add_argument('--fuse_qkv', action='store_true', default=False, help='Fuse QKV projections')
    parser.add_argument('--resume_qs', action='store_true', default=False, help='Resume SmoothQuant calibration')
    parser.add_argument('--use_gptq', action='store_true', default=False, help='Use GPTQ quantization')
    parser.add_argument('--resume_gptq', action='store_true', default=False, help='Resume GPTQ quantization')

    return parser.parse_args()


# 获得简单的校准集
def get_simple_calibration_data( device, min_num_images,num_frames,
    co3d_dir, co3d_anno_dir, SEEN_CATEGORIES, each_nsamples=5, cache_path=None):

    total_num = 0
    calib_data = []
    for category in SEEN_CATEGORIES:
        print(f"Loading calibration annotation for {category} test set")
        annotation_file = os.path.join(co3d_anno_dir, f"{category}_test.jgz")

        # 标签地址：/data2/fwl/datasets/co3d_v2_annotations/apple_test.jgz
        print(f"annotation_file: {annotation_file}")

        try:
            with gzip.open(annotation_file, "r") as fin:
                annotation = json.loads(fin.read())
                # print(f"annotation: {annotation}")
        except FileNotFoundError:
            print(f"Annotation file not found for {category}, skipping")
            continue

        seq_names = sorted(list(annotation.keys()))
        seq_names = random.sample(seq_names, min(each_nsamples, len(seq_names)))  # Random sample of sequences
        total_num += min(each_nsamples, len(seq_names))
        print(f"Processing Sequences: {seq_names}")

        for seq_name in seq_names:
            seq_data = annotation[seq_name]


            print("-" * 50)
            print(f"Processing {seq_name} for {category} test set")


            if len(seq_data) < min_num_images:  # Ensure sufficient data
                continue

            metadata = []
            for data in seq_data:
      
                if data["T"][0] + data["T"][1] + data["T"][2] > 1e5:
                    continue
                extri_opencv = convert_pt3d_RT_to_opencv(data["R"], data["T"])

                metadata.append({
                    "filepath": data["filepath"],
                    "extri": extri_opencv,
                })

            # Randomly sample num_frames images
            ids = np.random.choice(len(metadata), num_frames, replace=False) # 只是索引 
            image_names = [os.path.join(co3d_dir, metadata[i]["filepath"]) for i in ids]

            # Load and preprocess images
            images = load_and_preprocess_images(image_names , mode = 'pad').to(device)

            input_dict = {}
            input_dict["images"] = images
            input_dict["category"] = category # 新增类别
            input_dict["seq_name"] = seq_name # 每个类的id
            # ? model直接处理的是images
            calib_data.append(input_dict)
            # print(input_dict)
            print("len(input_dict[images])",len(input_dict["images"]))
 
            
    print("cache_path",{cache_path})
    if cache_path:
        print(f"Saving calibration data to: {cache_path}")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(calib_data, cache_path)

    print("⭐",total_num)
    return calib_data,len(calib_data)



def set_random_seeds(seed):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# 处理单个序列
def process_sequence(model, seq_name, seq_data, category, co3d_dir, min_num_images, num_frames, use_ba, device, dtype):
    """
    Process a single sequence and compute pose errors.

    Args:
        model: VGGT model
        seq_name: Sequence name
        seq_data: Sequence data
        category: Category name
        co3d_dir: CO3D dataset directory
        min_num_images: Minimum number of images required
        num_frames: Number of frames to sample
        use_ba: Whether to use bundle adjustment
        device: Device to run on
        dtype: Data type for model inference

    Returns:
        rError: Rotation errors
        tError: Translation errors
    """

    if len(seq_data) < min_num_images:
        return None, None

    metadata = []
    for data in seq_data:
        # Make sure translations are not ridiculous
        if data["T"][0] + data["T"][1] + data["T"][2] > 1e5:
            return None, None
        extri_opencv = convert_pt3d_RT_to_opencv(data["R"], data["T"])
        metadata.append({
            "filepath": data["filepath"],
            "extri": extri_opencv,
        })

    # Random sample num_frames images
    ids = np.random.choice(len(metadata), num_frames, replace=False)
    print("Image ids", ids)

    image_names = [os.path.join(co3d_dir, metadata[i]["filepath"]) for i in ids]
    gt_extri = [np.array(metadata[i]["extri"]) for i in ids]
    gt_extri = np.stack(gt_extri, axis=0)

    images = load_and_preprocess_images(image_names).to(device)

    # now_time = time.time()
    # time1 = now_time - start_time
    # print(f"time1:{time1}")

    if use_ba:
        try:
            pred_extrinsic = run_vggt_with_ba(model, images, image_names=image_names, dtype=dtype)
        except Exception as e:
            print(f"BA failed with error: {e}. Falling back to standard VGGT inference.")
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    predictions = model(images)
            with torch.cuda.amp.autocast(dtype=torch.float64):
                extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
                pred_extrinsic = extrinsic[0]
    else:
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                
                # now_time = time.time()
                predictions = model(images)
                # now_time2 = time.time()
                # forward_time = now_time2 - start_time

                # print(f"forward_time:{forward_time}")
        with torch.cuda.amp.autocast(dtype=torch.float64):
            extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
            pred_extrinsic = extrinsic[0]

    # now_time = time.time()
    # time2 = now_time - start_time
    # print(f"time2:{time2}")


    with torch.cuda.amp.autocast(dtype=torch.float64):
        gt_extrinsic = torch.from_numpy(gt_extri).to(device)
        add_row = torch.tensor([0, 0, 0, 1], device=device).expand(pred_extrinsic.size(0), 1, 4)

        pred_se3 = torch.cat((pred_extrinsic, add_row), dim=1)
        gt_se3 = torch.cat((gt_extrinsic, add_row), dim=1)

        # Set the coordinate of the first camera as the coordinate of the world
        # NOTE: DO NOT REMOVE THIS UNLESS YOU KNOW WHAT YOU ARE DOING
        pred_se3 = align_to_first_camera(pred_se3)
        gt_se3 = align_to_first_camera(gt_se3)

        rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(pred_se3, gt_se3, num_frames)

        Racc_5 = (rel_rangle_deg < 5).float().mean().item()
        Tacc_5 = (rel_tangle_deg < 5).float().mean().item()

        print(f"{category} sequence {seq_name} R_ACC@5: {Racc_5:.4f}")
        print(f"{category} sequence {seq_name} T_ACC@5: {Tacc_5:.4f}")

        # now_time = time.time()
        # time3 = now_time - start_timeq
        # print(f"time3:{time3}")
        # import pdb; pdb.set_trace() 
        return rel_rangle_deg.cpu().numpy(), rel_tangle_deg.cpu().numpy()
    


# 加载模型
def load_model(device, each_nsamples,min_num_images, num_frames,category,co3d_anno_dir,co3d_dir,
               model_path, dtype, compile, fuse_qkv, resume_qs, use_gptq, resume_gptq,debug_mode,
               not_smooth, not_rot, lwc,lac,rv,exp_name,cache_path = None):

    print("Initializing and loading VGGT model...")
    model = VGGT()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.to(device)

    calib_data = None

    if dtype in['kmeans']:
        if os.path.exists(cache_path) and not resume_qs:
            print(f"✅✅ cache_文件存在,加载中...: {cache_path}")
            calib_data = torch.load(cache_path)

        elif not os.path.exists(cache_path) and not resume_qs :
            print(f" ❌❌ cache_文件不存在,{each_nsamples}")
            calib_data, _ = get_simple_calibration_data(device, min_num_images,num_frames,
                                                co3d_dir,co3d_anno_dir, category, each_nsamples, cache_path=cache_path)
        if calib_data is not None:
            print("calib-data 已经正常加载")
        print("calib_data个数:",len(calib_data))
        # import pdb; pdb.set_trace() 
        return model,calib_data
    


def main():
    """Main function to evaluate VGGT on CO3D dataset."""
    # Parse command-line arguments
    args = setup_args()

    # Setup device and data type
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    

    # Categories to evaluate,类别
    SEEN_CATEGORIES = ["apple"]
    if args.debug_mode == "apple":
        SEEN_CATEGORIES = ["apple"]
    elif args.debug_mode == "five":
        SEEN_CATEGORIES = ["apple","bicycle", "bottle", "bowl","handbag"]
    elif args.debug_mode == "more":
        SEEN_CATEGORIES = ["apple","bicycle", "bottle", "bowl","handbag","carrot","cellphone", "motorcycle","umbrella","toaster"]
    elif args.debug_mode == "all":
        SEEN_CATEGORIES = [
        "apple", "backpack", "banana", "baseballbat", "baseballglove",
        "bench", "bicycle", "bottle", "bowl", "broccoli",
        "cake", "car", "carrot", "cellphone", "chair",
        "cup", "donut", "hairdryer", "handbag", "hydrant",
        "keyboard", "laptop", "microwave", "motorcycle", "mouse",
        "orange", "parkingmeter", "pizza", "plant", "stopsign",
        "teddybear", "toaster", "toilet", "toybus", "toyplane",
        "toytrain", "toytruck", "tv", "umbrella", "vase", "wineglass",
    ]

    # Load model
    model,calib_data= load_model(device,args.each_nsamples,
                       min_num_images =args.min_num_images, num_frames =  args.num_frames,category =SEEN_CATEGORIES,co3d_anno_dir = args.co3d_anno_dir,co3d_dir = args.co3d_dir, # 这两个参数用于calib_data
                        model_path=args.model_path, dtype=args.dtype,
                        compile= args.compile, fuse_qkv=args.fuse_qkv,
                        resume_qs=args.resume_qs, use_gptq=args.use_gptq,
                        resume_gptq=args.resume_gptq,debug_mode=args.debug_mode,
                        not_smooth=args.not_smooth, not_rot=args.not_rot,lac=args.lac,lwc=args.lwc,rv=args.rv,exp_name=args.exp_name,cache_path=args.cache_path)
    print("⭐ Load model done!")



    # import pdb; pdb.set_trace() 

    print(SEEN_CATEGORIES)
    # Set random seeds
    set_random_seeds(args.seed)
 
    print("⭐ 结束")
    return 


if __name__ == "__main__":
    print("OK")
    main()

