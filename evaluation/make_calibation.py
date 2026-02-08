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
from collections import defaultdict
from matplotlib import colors
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.pyplot as plt

logging.getLogger("dinov2").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="dinov2")

torch.set_float32_matmul_precision('highest')
torch.backends.cudnn.allow_tf32 = False

from sklearn.cluster import KMeans
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


def setup_args():
    """Set up command-line arguments for the CO3D evaluation script."""
    parser = argparse.ArgumentParser(description='Test VGGT on CO3D dataset')
    parser.add_argument('--use_ba', action='store_true', default=False, help='Enable bundle adjustment')
    parser.add_argument('--min_num_images', type=int, default=50, help='Minimum number of images for a sequence')
    parser.add_argument('--num_frames', type=int, default=10, help='Number of frames to use for testing')
    parser.add_argument('--co3d_dir', type=str, required=True, help='Path to CO3D dataset')
    parser.add_argument('--co3d_anno_dir', type=str, required=True, help='Path to CO3D annotations')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the VGGT model checkpoint')

    parser.add_argument('--kmeans_n', type=int, default=5, help='Number of cluster centers')
    parser.add_argument('--kmeans_m', type=int, default=4, help='Number of samples per cluster')

    parser.add_argument('--class_mode', type=str, default='all', help='Method for selecting calibration set categories')
    parser.add_argument('--each_nsamples', type=int, default=2, help='Number of samples per category to select')
    parser.add_argument('--cache_path', type=str, default=None, help='Path to load pre-filtered calibration set; creates new if not provided')

    parser.add_argument('--save_path', type=str, default='calib_data_final.pt', help='Path to save the calibration dataset')

    return parser.parse_args()


def get_simple_calibration_data( device, min_num_images,num_frames,
    co3d_dir, co3d_anno_dir, SEEN_CATEGORIES, each_nsamples=5, cache_path=None):

    total_num = 0
    calib_data = []
    for category in SEEN_CATEGORIES:
        print(f"Loading calibration annotation for {category} test set")
        annotation_file = os.path.join(co3d_anno_dir, f"{category}_test.jgz")
        print(f"annotation_file: {annotation_file}")

        try:
            with gzip.open(annotation_file, "r") as fin:
                annotation = json.loads(fin.read())
        except FileNotFoundError:
            print(f"Annotation file not found for {category}, skipping")
            continue

        seq_names = sorted(list(annotation.keys()))
        seq_names = random.sample(seq_names, min(each_nsamples, len(seq_names))) 
        total_num += min(each_nsamples, len(seq_names))
        print(f"Processing Sequences: {seq_names}")

        for seq_name in seq_names:
            seq_data = annotation[seq_name]
            print("-" * 50)
            print(f"Processing {seq_name} for {category} test set")
            if len(seq_data) < min_num_images:
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
            ids = np.random.choice(len(metadata), num_frames, replace=False)
            image_names = [os.path.join(co3d_dir, metadata[i]["filepath"]) for i in ids]
            images = load_and_preprocess_images(image_names , mode = 'pad').to(device)
            input_dict = {}
            input_dict["images"] = images
            input_dict["category"] = category 
            input_dict["seq_name"] = seq_name 
            calib_data.append(input_dict)
            print("len(input_dict[images])",len(input_dict["images"]))
      
    if cache_path:
        print(f"Saving calibration data to: {cache_path}")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(calib_data, cache_path)

    return calib_data,len(calib_data)


def filter_by_iqr(records, layer_names):
  
    category_to_indices = {}
    for idx, rec in enumerate(records):
        cat = rec["category"]
        category_to_indices.setdefault(cat, []).append(idx)

    abnormal_indices = set()

    for layer_name in layer_names:
        for cat, indices in category_to_indices.items():
            cat_means = np.array([records[i]['ln_info'][layer_name]['mean'] for i in indices])
            cat_stds  = np.array([records[i]['ln_info'][layer_name]['std']  for i in indices])

            if len(cat_means) < 4:
                continue

            q1_mean, q3_mean = np.percentile(cat_means, [25, 75])
            iqr_mean = q3_mean - q1_mean
            lower_mean, upper_mean = q1_mean - 1.5 * iqr_mean, q3_mean + 1.5 * iqr_mean

            q1_std, q3_std = np.percentile(cat_stds, [25, 75])
            iqr_std = q3_std - q1_std
            lower_std, upper_std = q1_std - 1.5 * iqr_std, q3_std + 1.5 * iqr_std

            for local_idx, global_idx in enumerate(indices):
                mean_val, std_val = cat_means[local_idx], cat_stds[local_idx]
                if not (lower_mean <= mean_val <= upper_mean and
                        lower_std  <= std_val  <= upper_std):
                    abnormal_indices.add(global_idx)

    all_indices = set(range(len(records)))
    kept_indices = sorted(all_indices - abnormal_indices)
    removed_indices = sorted(abnormal_indices)

    print(f"✅ keep {len(kept_indices)} 个")
    print(f"❌ delete {len(removed_indices)} 个")

    return all_indices,kept_indices, removed_indices


class LNFeatureAnalyzer:
    def __init__(self, model, dataloader, target_layer_names, dev):
        self.model = model
        self.dataloader = dataloader
        self.target_layer_names = target_layer_names  
        self.dev = dev
        self.hooks = []

        self.sample_records = []
        self.ln_cache = {}

    def _global_hook(self, module, input, output, layer_name):
        x = input[0]
        self.ln_cache[layer_name] = {
            'mean': x[0].mean().item(),
            'std': x[0].std().item(),
            'var': x[0].var().item(),
            'shape': tuple(x[0].shape)
        }

    def register_hooks(self):
        self.hooks.clear()
        for name, module in self.model.named_modules():
            if any(target in name for target in self.target_layer_names):
                hook = module.register_forward_hook(
                    lambda m, inp, out, n=name: self._global_hook(m, inp, out, n)
                )
                self.hooks.append(hook)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


    def analyze_and_save(self, save_path="sample_records2.pt", num_samples=None ):
        self.model.eval()
        self.sample_records.clear()

        with torch.no_grad():
            for idx, batch in enumerate(self.dataloader):
                if num_samples and len(self.sample_records) >= num_samples:
                    break
                self.ln_cache.clear()
                input_dict = {}
                for key in batch.keys():
                    if key == "images":
                        input_dict[key] = batch[key].to(self.dev) if torch.is_tensor(batch[key]) else batch[key]

                images = input_dict["images"]
                if len(images.shape) == 4:
                    images = images.unsqueeze(0)  

                output_list, _ = self.model(images)
                output_list = output_list[-1] 
                pooled_output_list = []

                for t in output_list:
                    t = t.squeeze(0)  
                    t = F.avg_pool1d(t, kernel_size=2, stride=2)

                    camera_token = t[:, :5, :] 
                    patch_token = t[:, 5:, :]  

                    patch_token = F.avg_pool1d(patch_token.permute(0, 2, 1), kernel_size=2, stride=2)  
                    patch_token = patch_token.permute(0, 2, 1) 

                    final_token = torch.cat([camera_token, patch_token], dim=1)
                    pooled_output_list.append(final_token.unsqueeze(0))

                record = {}
                record["output_list"] = [t[0].unsqueeze(0) for t in pooled_output_list]  
                record["category"] = batch["category"]
                record["seq_name"] = batch["seq_name"]

                record['ln_info'] = {}
                for layer_name, stats in self.ln_cache.items():
                    record['ln_info'][layer_name] = stats

                self.sample_records.append(record)

                torch.cuda.empty_cache()
                print(f"{idx+1}/{len(self.dataloader)}")

        torch.save(self.sample_records, save_path)
        print(f"save to: {save_path}")
        return self.sample_records



def Analysis_LN(model, dataloader, dev,
                ln_target_layers=None,
                kmeans_n_clusters = 1,
                kmeans_m_per_cluster = 1,
                final_subset_path="calib_data_final.pt"):

    if ln_target_layers is None:
        ln_target_layers = []
        for i in range(20, 24):
            ln_target_layers.append(f'frame_blocks.{i}.norm1')
            ln_target_layers.append(f'global_blocks.{i}.norm1')

    analyzer = LNFeatureAnalyzer(model=model,
                                 dataloader=dataloader,
                                 target_layer_names=ln_target_layers,
                                 dev=dev)

    analyzer.register_hooks()
    ln_results = analyzer.analyze_and_save()
    analyzer.remove_hooks()

    all_indices ,kept_indices, removed_indices = filter_by_iqr(ln_results, ln_target_layers)
    print(f"save samples/delete samples: {len(kept_indices)},{len(removed_indices)}")
    for rec in ln_results:
        rec.pop("ln_info", None)

    calib_data_ln = [ln_results [i] for i in all_indices]
    selected_samples_kmeans, _ = kmeans_select_samples(
        calib_data_ln,
        n_clusters=kmeans_n_clusters,
        m_per_cluster=kmeans_m_per_cluster,
    )

    del(calib_data_ln)
    final_calib_subset = create_calib_data_subset(
        dataloader,
        selected_samples_kmeans,
        new_path=final_subset_path
    )
  
    return final_calib_subset


def kmeans_select_samples(features_list, n_clusters=5, m_per_cluster=10):
    X_list = []
    meta_list = []  

    for feat in features_list:
        category = feat["category"]
        seq_name = feat["seq_name"]
        output_list = feat["output_list"]

        last_block_output = output_list[-1]  
        last_block_tensor = last_block_output.squeeze(0).cpu() 
        num_frames = last_block_tensor.shape[0] 
        first_frame = last_block_tensor[0]
        sim_list = []
        for frame_idx in range(1, num_frames):
            current_frame = last_block_tensor[frame_idx] 
            
            cos_sim = torch.nn.functional.cosine_similarity(
                current_frame, 
                first_frame,   
                dim=1         
            )
            
            mean_sim = cos_sim.mean().item()
            sim_list.append(mean_sim)
        
        sample_vector = np.array(sim_list)
        X_list.append(sample_vector)
        meta_list.append({"category": category, "seq_name": seq_name})

        del last_block_tensor, first_frame
        torch.cuda.empty_cache()

    X = np.stack(X_list, axis=0)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=50)
    cluster_ids = kmeans.fit_predict(X)

    cluster_counts = {}
    for cid in range(n_clusters):
        idx = np.where(cluster_ids == cid)[0]
        cluster_counts[cid] = len(idx)

    pca = PCA(n_components=3, random_state=42)
    X_3d = pca.fit_transform(X)
    explained_var = pca.explained_variance_ratio_

    categories = list(set([meta["category"] for meta in meta_list]))
    n_categories = len(categories)
    category_colors = plt.cm.Set3(np.linspace(0, 1, n_categories))

    selected_samples = []
    selected_indices = [] 
    all_indices = list(range(len(meta_list)))

    for cid in range(n_clusters):
        idx = np.where(cluster_ids == cid)[0].tolist()
        if len(idx) > m_per_cluster:
            keep = random.sample(idx, m_per_cluster)
        else:
            keep = idx 
        
        selected_indices.extend(keep)
        for i in keep:
            selected_samples.append(meta_list[i])

    total_needed = n_clusters * m_per_cluster
    current_count = len(selected_samples)
    additional_needed = total_needed - current_count

    if additional_needed > 0:
        unselected_indices = [i for i in all_indices if i not in selected_indices]
        
        if len(unselected_indices) >= additional_needed:
            additional_keep = random.sample(unselected_indices, additional_needed)
            for i in additional_keep:
                selected_samples.append(meta_list[i])
        else:
            for i in unselected_indices:
                selected_samples.append(meta_list[i])

    print(f"Total samples saved: {len(selected_samples)} / {len(features_list)}")
    
    cluster_info = {
        'X_3d': X_3d,
        'cluster_ids': cluster_ids,
        'explained_variance': explained_var,
        'kmeans_model': kmeans,
        'categories': categories,
        'category_colors': category_colors
    }
    
    return selected_samples, cluster_info

def create_calib_data_subset(calib_data, selected_samples, new_path=None):

    category_to_samples = defaultdict(dict)
    for sample in calib_data:
        category = sample["category"]
        seq_name = sample["seq_name"]
        category_to_samples[category][seq_name] = sample

    selected_calib_data = []
    for s in selected_samples:
        category = s["category"]
        seq_name = s["seq_name"]

        if category in category_to_samples and seq_name in category_to_samples[category]:
            selected_calib_data.append(category_to_samples[category][seq_name])
        else:
            print(f"Warning: ({category}, {seq_name}) not found in original calib_data")

    if new_path:
        torch.save(selected_calib_data, new_path)
    else:
        print(f"{new_path} not find")
    
    print(f"New calibation dataset has been saved to : {new_path}")
    return selected_calib_data


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
    


def load_model(device, each_nsamples,min_num_images, num_frames,category,co3d_anno_dir,co3d_dir,
               model_path,cache_path = None):

    print("Initializing and loading VGGT model...")
    model = VGGT()
    model.load_state_dict(torch.load(model_path))
    calib_data = None

    if os.path.exists(cache_path) :
        print(f"loading ...: {cache_path}")
        calib_data = torch.load(cache_path)

    elif not os.path.exists(cache_path) :
        print(f"{cache_path} not exist")
        calib_data, _ = get_simple_calibration_data(device, min_num_images,num_frames,
                                            co3d_dir,co3d_anno_dir, category, each_nsamples, cache_path=cache_path)
    if calib_data is None:
        print("Error")
 
    return model,calib_data
    
def main():
    """Main function to evaluate VGGT on CO3D dataset."""
    args = setup_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    SEEN_CATEGORIES = ["apple"]
    if args.class_mode == "apple":
        SEEN_CATEGORIES = ["apple"]
    elif args.class_mode == "five":
        SEEN_CATEGORIES = ["apple","bicycle", "bottle", "bowl","handbag"]
    elif args.class_mode == "more":
        SEEN_CATEGORIES = ["apple","bicycle", "bottle", "bowl","handbag","carrot","cellphone", "motorcycle","umbrella","toaster"]
    elif args.class_mode == "all":
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

    model,calib_data= load_model(device,args.each_nsamples,
                       min_num_images =args.min_num_images, num_frames =  args.num_frames,
                       category =SEEN_CATEGORIES,co3d_anno_dir = args.co3d_anno_dir,co3d_dir = args.co3d_dir,
                        model_path=args.model_path,cache_path=args.cache_path)
 
    aggregator = model.aggregator.to(device)
    Analysis_LN(aggregator, calib_data, device,
                ln_target_layers=None,
                kmeans_n_clusters=args.kmeans_n, 
                kmeans_m_per_cluster=args.kmeans_m, 
                final_subset_path=args.save_path)
    
    return 


if __name__ == "__main__":
    main()

