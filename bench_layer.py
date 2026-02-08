from vggt.layers.mlp import Mlp
from vggt.layers.block import Block
import torch
import time
from torchao.quantization import Int8DynamicActivationInt8WeightConfig, Int8DynamicActivationInt4WeightConfig, Int4WeightOnlyConfig, Int8WeightOnlyConfig, quantize_, MappingType
from torchao.utils import (
    benchmark_model,
)
import copy

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}") 

# test_layer = Mlp(1024, 4096, 1024).eval().to(torch.bfloat16).to(device)
test_layer = Block(1024, 16).eval().to(torch.bfloat16).to(device)

B, S, P, C = 1, 40, 930, 1024
tokens = ((torch.randn(B, P*S, C).to(device).to(torch.bfloat16)), )
test_layer_fp = copy.deepcopy(test_layer)
warm_up = 50
# quantize_(test_layer, Int8DynamicActivationInt8WeightConfig())
quantize_(test_layer, Int8DynamicActivationInt4WeightConfig(group_size=1024, act_mapping_type=MappingType.SYMMETRIC))
# quantize_(test_layer, Int8WeightOnlyConfig())
test_layer.forward = torch.compile(test_layer.forward, mode="reduce-overhead", fullgraph=True)
bench_time = 100
for _ in range(warm_up):
    test_layer(tokens[0])
int8_time = benchmark_model(test_layer, bench_time, tokens)
print("int8 mean time: %0.3f ms" % int8_time)

test_layer_fp = torch.compile(test_layer_fp, mode="max-autotune", fullgraph=True)
prefill_time = 100
for _ in range(warm_up):
    test_layer_fp(tokens[0])
fp_time = benchmark_model(test_layer_fp, prefill_time, tokens)
print("fp mean time: %0.3f ms" % fp_time)