from .quarot_linear import SmoothRotQuantizedLinear
from .train_utils import cali_qs_quant
from .function_utils import get_paras_dict_by_name
import torch.nn as nn
import torch
import logging
import os
import math
import json
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

def load_qs_parameters(args, model, path=None):
    print(".........loading",args.exp_dir)
    if path is None:
        absolute_path = os.path.abspath(os.path.join(args.exp_dir, f"qs_frame_parameters_total.pth"))
        print(absolute_path)
        qs_frame_parameters = torch.load(os.path.join(args.exp_dir, f"qs_frame_parameters_total.pth"))
        qs_global_parameters = torch.load(os.path.join(args.exp_dir, f"qs_global_parameters_total.pth"))
    else:
        qs_frame_parameters = torch.load(os.path.join(path, f"qs_frame_parameters_total.pth"))
        qs_global_parameters = torch.load(os.path.join(path, f"qs_global_parameters_total.pth"))

    frame_layers = model.aggregator.frame_blocks
    global_layers = model.aggregator.global_blocks


    for i in range(len(qs_frame_parameters.keys())):
        qs_frame_param = qs_frame_parameters[i]
        _, _ = frame_layers[i].load_state_dict(qs_frame_param, strict=False)
       
    for i in range(len(qs_global_parameters.keys())):
        qs_global_param = qs_global_parameters[i]
        global_layers[i].load_state_dict(qs_global_param, strict=False)

    return model


def mark_ignore(module,ignore_quantize):
    for child in module.children():
        if isinstance(child, torch.nn.Linear):
            child.ignore_quantize = ignore_quantize
        else:
            mark_ignore(child,ignore_quantize)


def set_ignore_quantize(model, ignore_quantize=True):
    model.aggregator.patch_embed.patch_embed.ignore_quantize = True
    model.aggregator.patch_embed.norm.ignore_quantize = True
    model.aggregator.patch_embed.head.ignore_quantize = True

    for head in [model.aggregator.patch_embed.blocks ,
                 model.camera_head, 
                 model.point_head,
                 model.depth_head, 
                 model.track_head]:
        mark_ignore(head,ignore_quantize)
  

def is_power_of_two(n):
    return (n & (n - 1)) == 0 and n != 0


def check_linear_dims(model):
    non_power_of_two = defaultdict(list) 
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            in_features = module.in_features  
            out_features = module.out_features 
            
            if not is_power_of_two(in_features):
                non_power_of_two[in_features].append({
                    "layer_name": name,
                    "in_features": in_features,
                    "out_features": out_features
                })
    
    return non_power_of_two

def get_quantized_linear_layers(model):
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, SmoothRotQuantizedLinear):
            in_features = getattr(module.linear, "in_features", None)
            out_features = getattr(module.linear, "out_features", None)
            w_bits = getattr(module, "w_bits", None)
            a_bits = getattr(module, "a_bits", None)
            if w_bits is None:
                w_bits = getattr(getattr(module, "weight_quantizer", None), "bits", None)
            if a_bits is None:
                a_bits = getattr(getattr(module, "act_quantizer", None), "bits", None)

            layers.append({
                "name": name,
                "in_features": in_features,
                "out_features": out_features,
                "w_bits": w_bits,
                "a_bits": a_bits,
                "alpha": getattr(module, "smooth_quant_alpha", None),
                "higher_bits": getattr(module, "higher_bits", False),
            })
    return layers


def print_quantized_linear_layers(model, logger: logging.Logger | None = None):
    infos = get_quantized_linear_layers(model)
    if logger is None:
        logger = logging.getLogger("quarot")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            logger.addHandler(logging.StreamHandler())

    if not infos:
        return

    for i, info in enumerate(infos):
        logger.info(
            f"[{i}] {info['name']} in={info['in_features']} out={info['out_features']} "
            f"w_bits={info['w_bits']} a_bits={info['a_bits']} alpha={info['alpha']} higher_bits={info['higher_bits']}"
        )


def get_ignored_linear_layers(model):
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and getattr(module, "ignore_quantize", False):
            layers.append({
                "name": name,
                "in_features": module.in_features,
                "out_features": module.out_features,
            })
    return layers


def print_ignored_linear_layers(model, logger: logging.Logger | None = None):
    infos = get_ignored_linear_layers(model)
    if logger is None:
        logger = logging.getLogger("quarot")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            logger.addHandler(logging.StreamHandler())

    if not infos:
        return

    for i, info in enumerate(infos):
        logger.info(
            f"[{i}] {info['name']} in={info['in_features']} out={info['out_features']}"
        )


def save_quantized_linear_layers(model, out_dir: str, filename: str = "quantized_linear_layers.json"):
    infos = get_quantized_linear_layers(model)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(infos, f, ensure_ascii=False, indent=2)
    return out_path


def save_ignored_linear_layers(model, out_dir: str, filename: str = "ignored_linear_layers.json"):
    infos = get_ignored_linear_layers(model)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(infos, f, ensure_ascii=False, indent=2)
    return out_path


def quantize_linear(module, device="cuda", args=None):
    if isinstance(module, nn.Linear):
        use_rot = True
        if device is not None:
            module = module.to(device)

        if getattr(module, 'ignore_quantize', False):
            return module

        if getattr(module, 'higher_bits', False):
            original_a_bits = args.a_bits
            original_w_bits = args.w_bits
            args.w_bits = 8
            args.a_bits = 8
            if use_rot:
                new_layer = SmoothRotQuantizedLinear(args, module)
            args.a_bits = original_a_bits
            args.w_bits = original_w_bits
        else:
            if use_rot:
                new_layer = SmoothRotQuantizedLinear(args, module)
     
        return new_layer
    else:
        for name, child in module.named_children():
            new_child = quantize_linear(
                child, device, args
            )
            if new_child is not child:
                setattr(module, name, new_child)
        if device is not None:
            module.to(device=device) 
        return module


def save_hadamard_matrix(model, path):
    model_params = get_paras_dict_by_name(model, required_names=["rotation_matrix"])
    torch.save(model_params, os.path.join(path, f"hadamard_matrix.pth"))

def model_reparameterize(model):
    for name, module in model.named_modules():
        if isinstance(module,SmoothRotQuantizedLinear):
            module.reparameterize()

def init_logger(log_dir="logs", log_file="app.log"):
    logger = logging.getLogger("my_app")
    logger.setLevel(logging.DEBUG) 
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def quarot_smooth_quant_model(config,model,calib_data, wbit, abit, resume_qs=False,
                              use_gptq=False, resume_gptq=False, model_id=None,exp_name=None):
    
    device = next(model.parameters()).device
    model.to(device)
    set_ignore_quantize(model)
    quantize_linear(model, args=config)

    if resume_qs:
        model = load_qs_parameters(config, model)
        after_resume_qs(model)
        try:
            q_path = save_quantized_linear_layers(model, config.exp_dir)
            ig_path = save_ignored_linear_layers(model, config.exp_dir)
        except Exception as e:
            pass
        return 
    
    if not config.not_smooth :
        with torch.no_grad():
            num_calib = max(1, math.floor(config.nsamples / 5))
            for idx, batch in enumerate(calib_data[:num_calib]):
               
                input_dict = {}
                for key in batch.keys():
                    if key == "images":
                        if torch.is_tensor(batch[key]):
                            input_dict[key] = batch[key].to(device)
                        else:
                            input_dict[key] = batch[key]
            
                _ = model(**input_dict)
              
        after_smoothfactor_init(model)
 
    if not config.not_smooth or config.lac or config.lwc:
        logger = init_logger(config.exp_dir)
        cali_qs_quant(config, model, calib_data, device, logger)

    else:
        print("Do not need calib")

    after_resume_qs(model)
    try:
        q_path = save_quantized_linear_layers(model, config.exp_dir)
        ig_path = save_ignored_linear_layers(model, config.exp_dir)
    except Exception as e:
        pass
    return


def after_smoothfactor_init(model):
    for name, module in model.named_modules():
        if isinstance(module, (SmoothRotQuantizedLinear)):
            module.smooth_quant_running_stat = False
            module.ori_mode = True
            module.train_mode = False
            module.eval_mode = False

            module.channel_wise_scale = nn.Parameter(module.act_quantizer.act_scale.pow(module.smooth_quant_alpha) / module.linear.weight.abs().max(dim=0)[0].pow(1 - module.smooth_quant_alpha))
            module.channel_wise_scale.data = torch.clamp(module.channel_wise_scale.data, min=1e-5)


def after_resume_qs(model):
    i=0
    for name, module in model.named_modules():
        if isinstance(module, (SmoothRotQuantizedLinear)):
            module.smooth_quant_running_stat = False
            module.ori_mode = False
            module.train_mode = False
            module.eval_mode = True
            module.reparameterize() 

def enable_repq(model):
    for module in model.modules():
        if isinstance(module, SmoothRotQuantizedLinear):
            module.enable_calibration()

def after_repq(model):
    for module in model.modules():
        if isinstance(module, SmoothRotQuantizedLinear):
            module.enable_quantization()