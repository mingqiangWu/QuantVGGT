import os
import time
import gc
import functools
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from collections import OrderedDict
from diffusers.models.attention import Attention
from diffusers.models.attention_processor import CogVideoXAttnProcessor2_0


from .function_utils import set_require_grad_all, get_n_set_parameters_byname, get_paras_dict_by_name, check_params_grad
from .quant_utils import set_quantizer_state
from .quarot_linear import  SmoothRotQuantizedLinear

save_gpu_memory = True

def set_embed_to_device(model, device):
    model.aggregator.patch_embed.to(device)
    model.aggregator.rope.to(device)


def set_aggregator_to_ori(aggregator):
    set_linear_to_ori(aggregator) 
    for block in aggregator.frame_blocks:
        set_linear_to_ori(block) 
    for block in aggregator.global_blocks:
        set_linear_to_ori(block)



def set_aggregator_to_train(aggregator):
    set_linear_to_train(aggregator)
    for block in aggregator.frame_blocks:
        set_linear_to_train(block)
    for block in aggregator.global_blocks:
        set_linear_to_train(block)


def set_aggregator_to_eval(aggregator):
    set_linear_to_eval(aggregator)
    for block in aggregator.frame_blocks:
        set_linear_to_eval(block) 
    for block in aggregator.global_blocks:
        set_linear_to_eval(block) 


def set_linear_to_ori(layer):
    for name, module in layer.named_modules():
        if isinstance(module, (SmoothRotQuantizedLinear)):
            module.ori_mode = True
            module.smooth_quant_running_stat = False
            module.train_mode = False 
            module.eval_mode = False

def set_linear_to_train(layer):
    for name, module in layer.named_modules():
        if isinstance(module, (SmoothRotQuantizedLinear)):
            module.ori_mode = False
            module.train_mode = True
            module.eval_mode = False

def set_linear_to_eval(layer):
    for name, module in layer.named_modules():
        if isinstance(module, (SmoothRotQuantizedLinear)):
            module.ori_mode = False
            module.train_mode = False
            module.eval_mode = True


def cali_qs_quant_agg(args, model, dataloader, dev, logger):
    def print_memory_stats(prefix=""):
        logger.info(f"{prefix} Memory Stats:")
        logger.info(f"Allocated: {torch.cuda.memory_allocated(dev) / 1024 ** 2:.2f} MB")
        logger.info(f"Reserved/Cached: {torch.cuda.memory_reserved(dev) / 1024 ** 2:.2f} MB")
        logger.info(f"Max Allocated: {torch.cuda.max_memory_allocated(dev) / 1024 ** 2:.2f} MB")
        logger.info(f"Max Reserved: {torch.cuda.max_memory_reserved(dev) / 1024 ** 2:.2f} MB")

    print_memory_stats("Initial")
    model.eval()

    for name, param in model.named_parameters():
        param.requires_grad = False

    args.deactive_amp = True
    if args.deactive_amp:
        dtype = torch.float32
        traincast = nullcontext
    else:
        dtype = torch.bfloat16
        dtype = torch.float32
        traincast = functools.partial(torch.amp.autocast, device_type="cuda", dtype=dtype)


    aggregator = model.aggregator
    frame_blocks = aggregator.frame_blocks
    global_blocks = aggregator.global_blocks

    last_layer = global_blocks[23].to(dev)
    set_embed_to_device(model, dev)

    first_ips = {
        "tokens": [],
    }
    last_outs = {
        "tokens": [],
    }
    cache = {"i": 0,
             "pos": []}

    class Catcher_ips(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, x, pos):
            if save_gpu_memory:
                first_ips["tokens"].append(x.to("cpu"))  # to tuple
            else:
                first_ips["tokens"].append(x)  # to tuple
            cache["i"] += 1
            cache["pos"].append(pos) 

            return self.module(x,pos)

    class Catcher_outs(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, x, pos):
            outs = self.module(x,pos)
            if save_gpu_memory:
                last_outs["tokens"].append(outs.to("cpu"))  
            else:
                last_outs["tokens"].append(outs)

            raise ValueError
        
    set_aggregator_to_ori(aggregator) 
    model.aggregator.global_blocks[23] = Catcher_outs(last_layer)    

    with torch.no_grad():
        for batch in dataloader:
            try:
                input_dict = {}
                for key in batch.keys():
                    if key == "images":
                        if torch.is_tensor(batch[key]):
                            input_dict[key] = batch[key].to(dev)
                        else:
                            input_dict[key] = batch[key]
                model(**input_dict)
            except ValueError:
                pass

    model.aggregator.global_blocks[23] = model.aggregator.global_blocks[23].module
    set_aggregator_to_train(aggregator) 
    torch.cuda.empty_cache()
    for name, param in model.named_parameters():
        param.requires_grad = True
    loss_func = torch.nn.MSELoss()

    def calib_layer(layer, idx, block_name, fp_inps, fp_outs):
        logger.info(f"========= {block_name}_Layer_{idx} =========")
        print_memory_stats(f"Before {block_name}_Layer_{idx}")
        dtype_dict = {}
        layer = layer.to(dev)
        for name, param in layer.named_parameters():
            dtype_dict[name] = param.dtype
        with torch.no_grad():
            layer.float()

        set_linear_to_ori(layer)

        with torch.no_grad():
            for j in range(args.nsamples):    
                fp_tokens = layer( x = fp_inps["tokens"][j].float().to(dev),
                                   pos = cache["pos"][j].to(dev))
                fp_outs["tokens"].append(fp_tokens.to("cpu"))
                del fp_tokens
                torch.cuda.empty_cache()

        set_linear_to_train(layer) 

        set_require_grad_all(layer, False)
        trained_params, paras_name = [], []

        if not args.not_smooth:
            trained_params.append(
                {"params": get_n_set_parameters_byname(layer, ["channel_wise_scale"]),"lr": args.qs_lr,"name": "channel_wise_scale"})
            paras_name.append("channel_wise_scale")
        
        if not args.not_rot:
            paras_name.append("rotation_matrix")
      
        if args.lwc:
            trained_params.append(
                {"params": get_n_set_parameters_byname(layer, ["clip_factor_w"]),"lr": args.qs_lr * 10,"name": "clip_factor_w"})
            paras_name.append("clip_factor_w")

        if args.lac:
            trained_params.append(
                {"params": get_n_set_parameters_byname(layer, ["clip_factor_a"]),"lr": args.qs_lr * 10,"name": "clip_factor_a"})
            paras_name.append("clip_factor_a")
        
        if True:
            trained_params.append(
                {"params": get_n_set_parameters_byname(layer, ["adaround_alpha", "adaround_beta"]),"lr": args.qs_lr * 10,"name": "ada_param",
                "lr": args.qs_lr,
                "name": "adaround_params"
            })

        optimizer = torch.optim.AdamW(trained_params)
        scheduler_main = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * args.nsamples,
                                                                    eta_min=args.qs_lr * 1e-3)
        if args.warmup:
            scheduler_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=16)
            scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler_warmup, scheduler_main])
        else:
            scheduler = scheduler_main
        
        for epoch in range(args.epochs):
            if epoch == 0:
                print_memory_stats(f"Layer {idx} Epoch {epoch} Start")
            mse = 0
            start_tick = time.time()
            with traincast():
                for j in range(args.nsamples):
                    cur_tokens = fp_inps["tokens"][j].to(dev)
                    cur_fp_tokens= fp_outs["tokens"][j].to(dev)
                    quant_tokens= layer(
                        x = cur_tokens,
                        pos=cache["pos"][j],
                    )

                    loss = loss_func(cur_fp_tokens, quant_tokens)
                    mse += loss.detach().cpu()
                    optimizer.zero_grad()
                    loss.backward()

                    optimizer.step()
                    scheduler.step()

                    del cur_tokens,cur_fp_tokens,quant_tokens
                    torch.cuda.empty_cache()

            cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
            logger.info(
                f"layer:[{idx}] , lwc lac iter:[{epoch}], lr:[{cur_lr:.8f}] ,time:[{time.time() - start_tick:.6f}]s, mse: [{mse / args.nsamples:.8f}]")
        
        fp_inps = fp_outs
        fp_outs = {
            "tokens": [],
        }
        layer= layer.to("cpu")
        layer_params = get_paras_dict_by_name(layer, required_names=paras_name)
        torch.save(layer_params, os.path.join(args.exp_dir, f"qs_{block_name}_parameters_layer_{idx}.pth"))
        logger.info(
            f"saved parameters for layer {idx} at {os.path.join(args.exp_dir, f'qs_{block_name}_parameters_layer_{idx}.pth')}")
       
        del layer_params  
        for name, param in layer.named_parameters():
            param.requires_grad = False
            if name in dtype_dict.keys():
                param.data = param.to(dtype_dict[name]) 

        del layer
        torch.cuda.empty_cache()
        print_memory_stats(f"After {block_name}_Layer_{idx}")
        return fp_inps,fp_outs

    block_num = model.aggregator.aa_block_num
    aa_order = model.aggregator.aa_order

    for idx in range(block_num):
        for attn_type in aa_order:
            if attn_type == "frame":
                fp_inps, fp_outs = calib_layer(frame_blocks[idx], idx, "frame", fp_inps, fp_outs)
                frame_blocks[idx] = frame_blocks[idx].to("cpu")
                for param in frame_blocks[idx].parameters():
                        param.requires_grad = False
             
                
            elif attn_type == "global":
                fp_inps, fp_outs = calib_layer(global_blocks[idx], idx, "global", fp_inps, fp_outs)
                global_blocks[idx] = global_blocks[idx].to("cpu")
                for param in global_blocks[idx].parameters():
                    param.requires_grad = False
             

    del fp_inps, fp_outs
    gc.collect()
    torch.cuda.empty_cache()
    
    model.eval()
    print_memory_stats("Final")
    model.to(dev)

    qs_frame_parameters = {}
    qs_global_parameters = {}

    for i in range(block_num):
        layer_frame_params = torch.load(os.path.join(args.exp_dir, f"qs_frame_parameters_layer_{i}.pth"))
        layer_global_params = torch.load(os.path.join(args.exp_dir, f"qs_global_parameters_layer_{i}.pth"))
        qs_frame_parameters[i] = layer_frame_params
        qs_global_parameters[i] = layer_global_params
        del layer_frame_params
        del layer_global_params
    torch.save(qs_frame_parameters, os.path.join(args.exp_dir, "qs_frame_parameters_total.pth"))
    torch.save(qs_global_parameters, os.path.join(args.exp_dir, f"qs_global_parameters_total.pth"))
    logger.info(f"saved merged parameters at {os.path.join(args.exp_dir, 'qs_frame_parameters_total.pth')}")
    logger.info(f"saved merged parameters at {os.path.join(args.exp_dir, 'qs_global_parameters_total.pth')}")


    for i in range(block_num):
        layer_frame_param_path = os.path.join(args.exp_dir, f"qs_frame_parameters_layer_{i}.pth")
        layer_global_param_path = os.path.join(args.exp_dir, f"qs_global_parameters_layer_{i}.pth")
        if os.path.exists(layer_frame_param_path):
            os.remove(layer_frame_param_path)
            logger.info(f"removed layer parameter file: {layer_frame_param_path}")
        if os.path.exists(layer_global_param_path):
            os.remove(layer_global_param_path)
            logger.info(f"removed layer parameter file: {layer_global_param_path}")

    return
