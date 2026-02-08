# -*- coding: utf-8 -*-
import torch
from collections import OrderedDict

def remove_rotation_matrix(model_dicts):
    """从所有层中删除rotation_matrix键"""
    cleaned_dicts = {}
    
    for layer_idx, layer_dict in model_dicts.items():
        # 对每一层应用删除操作
        cleaned_dict = OrderedDict()
        
        for key, value in layer_dict.items():
            if '.rotation_matrix' not in key:
                cleaned_dict[key] = value
        
        cleaned_dicts[layer_idx] = cleaned_dict
        
        # 打印每层的处理信息
        original_keys = len(layer_dict)
        cleaned_keys = len(cleaned_dict)
        removed_count = original_keys - cleaned_keys
        print(f"层 {layer_idx}: 原始 {original_keys} 个键 -> 删除 {removed_count} 个 -> 剩余 {cleaned_keys} 个键")
    
    return cleaned_dicts


# def remove_rotation_matrix(model_dicts):
#     """将所有层的rotation_matrix参数置为同等形状的全1向量"""
#     modified_dicts = {}
    
#     for layer_idx, layer_dict in model_dicts.items():
#         # 创建新字典，保持所有键
#         modified_dict = OrderedDict()
        
#         rotation_matrix_count = 0
#         total_keys = 0
        
#         for key, value in layer_dict.items():
#             total_keys += 1
#             if '.rotation_matrix' in key:
#                 # 将rotation_matrix置为同等形状的全1向量
#                 if isinstance(value, torch.Tensor):
#                     modified_value = torch.ones_like(value)
#                     rotation_matrix_count += 1
#                     print(f"层 {layer_idx}: 将 {key} 置为全1向量 (形状: {value.shape})")
#                 else:
#                     modified_value = value
#                     print(f"层 {layer_idx}: {key} 不是张量，保持原值")
#             else:
#                 # 保持其他参数不变
#                 modified_value = value
            
#             modified_dict[key] = modified_value
        
#         modified_dicts[layer_idx] = modified_dict
        
#         # 打印每层的处理信息
#         print(f"层 {layer_idx}: 总共 {total_keys} 个键 -> 修改 {rotation_matrix_count} 个rotation_matrix -> 保持 {total_keys} 个键")

#     return modified_dicts



# def remove_rotation_matrix(model_dicts):
#     """将所有层的rotation_matrix参数复制一次再写入同样的位置"""
#     modified_dicts = {}
    
#     for layer_idx, layer_dict in model_dicts.items():
#         # 创建新字典，保持所有键
#         modified_dict = OrderedDict()
        
#         rotation_matrix_count = 0
#         total_keys = 0
        
#         for key, value in layer_dict.items():
#             total_keys += 1
#             if '.rotation_matrix' in key:
#                 # 复制rotation_matrix并写入同样位置
#                 if isinstance(value, torch.Tensor):
#                     # 深拷贝张量
#                     duplicated_value = value.clone().detach()
#                     # duplicated_value = torch.ones_like(value)
#                     rotation_matrix_count += 1
#                     print(f"层 {layer_idx}: 复制 {key} (形状: {value.shape}) -> 写入同样位置")
#                 else:
#                     # 非张量，直接赋值
#                     duplicated_value = value
#                     print(f"层 {layer_idx}: {key} 不是张量，直接复制")
#             else:
#                 # 保持其他参数不变
#                 duplicated_value = value
            
#             modified_dict[key] = duplicated_value
        
#         modified_dicts[layer_idx] = modified_dict
        
#         # 打印每层的处理信息
#         print(f"层 {layer_idx}: 总共 {total_keys} 个键 -> 复制 {rotation_matrix_count} 个rotation_matrix -> 保持 {total_keys} 个键")

#     return modified_dicts

def load_pt_file(file_path):
    data = torch.load(file_path, map_location='cpu')
    print(data.keys())
    print(data[1].keys())
    print("每一层总参数",len(data[1].keys()))
    print(data[1])
    # print(data)  # 直接打印


if __name__ == "__main__":
        
    # 使用示例
    state_dict = torch.load("/data2/fwl/VGGT_quant/vggt/evaluation/outputs/w4a4/44_end_qs_2_model_tracker_fixed_e20.pt_sym/qs_frame_parameters_total.pth", map_location="cpu")
    new_state_dict = remove_rotation_matrix(state_dict)
    torch.save(new_state_dict, "/data2/fwl/VGGT_quant/vggt/evaluation/outputs/w4a4/test2_model_tracker_fixed_e20.pt_sym/qs_frame_parameters_total.pth")

    state_dict = torch.load("/data2/fwl/VGGT_quant/vggt/evaluation/outputs/w4a4/44_end_qs_2_model_tracker_fixed_e20.pt_sym/qs_global_parameters_total.pth", map_location="cpu")
    new_state_dict = remove_rotation_matrix(state_dict)
    torch.save(new_state_dict, "/data2/fwl/VGGT_quant/vggt/evaluation/outputs/w4a4/test2_model_tracker_fixed_e20.pt_sym/qs_global_parameters_total.pth")


    load_pt_file("/data2/fwl/VGGT_quant/vggt/evaluation/outputs/w4a4/44_end_qs_2_model_tracker_fixed_e20.pt_sym/qs_global_parameters_total.pth")
    load_pt_file("/data2/fwl/VGGT_quant/vggt/evaluation/outputs/w4a4/test2_model_tracker_fixed_e20.pt_sym/qs_global_parameters_total.pth")


    # load_pt_file("/data2/fwl/VGGT_quant/vggt/evaluation/outputs/w4a4/44_end_qs_2_model_tracker_fixed_e20.pt_sym/qs_frame_parameters_total.pth")
    # load_pt_file("/data2/fwl/VGGT_quant/vggt/evaluation/outputs/w4a4/test2_model_tracker_fixed_e20.pt_sym/qs_frame_parameters_total.pth")

