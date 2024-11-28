import os
import io
from classification.utils import *
from classification.ptflops import get_model_complexity_info
from classification.ptflops.pytorch_ops import linear_flops_counter_hook
from classification.ptflops.pytorch_ops import conv_flops_counter_hook
from classification.train import train_one_epoch
from neural_networks.CIFAR10.resnet import resnet8, resnet20, resnet32, resnet56
device = 'cuda'


#load dataset
val_data, calib_data = cifar10_data_loader(data_path="/home/michael/thesis_fw/data/", batch_size=128)



#select pretrained model
# an example repo with cifar10 models. you can use your own (ref: https://github.com/chenyaofo/pytorch-cifar-models)
#model = torch.hub.load("chenyaofo/pytorch-cifar-models", 'cifar10_repvgg_a0', pretrained=True).to(device)
mode= {"execution_type": 'float', "act_bit": 8, "weight_bit": 8, "bias_bit": 32, "fake_quant": True, "classes": 10, "act_type": 'ReLU'}
model = resnet8(mode).to(device)
filename = "examples/models/resnet8_a8_w8_b32_fake_quant_cifar10_ReLU.pth"
checkpoint = torch.load(filename, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'], strict=True)
model.to(device)

#optional: evaluate default model
top1 = evaluate_cifar10(model, val_data, device = device)



#initialize model with axx layers
# get conv2d layers to approximate
conv2d_layers = [(name, module) for name, module in model.named_modules() if (isinstance(module, torch.nn.Conv2d) or isinstance(module, AdaptConv2D)) and ("head" not in name and "reduction" not in name)]
print(f'type of conv2d_layers = {type(conv2d_layers)}')
len(conv2d_layers)

# Initialize model with all required approximate multipliers for axx layers. 
# No explicit assignment needed; this step JIT compiles all upcoming multipliers

axx_list = [{'axx_mult' : 'mul8s_acc', 'axx_power' : 1.0, 'quant_bits' : 8, 'fake_quant' : False}]*len(conv2d_layers)
axx_list[3:4] = [{'axx_mult' : 'bw_mult_9_9_0', 'axx_power' : 0.7082, 'quant_bits' : 8, 'fake_quant' : False}] * 1

start = time.time()
replace_conv_layers(model,  AdaptConv2D, axx_list, 0, 0, layer_count=[0], returned_power = [0], initialize = True)  
print('Time to compile cuda extensions: ', time.time()-start)



# measure flops of model and compute 'flops' in every layer
#hook our custom axx_layers in the appropriate flop counters, i.e. AdaptConv2D : conv_flops_counter_hook
with torch.cuda.device(0):
    total_macs, total_params, layer_specs = get_model_complexity_info(model, (3, 32, 32),as_strings=False, print_per_layer_stat=True,
                                                          custom_modules_hooks={AdaptConv2D : conv_flops_counter_hook}, 
                                                          param_units='M', flops_units='MMac',
                                                          verbose=True)

print(f'Computational complexity:  {total_macs/1000000:.2f} MMacs')
print(f'Number of parameters::  {total_params/1000000:.2f} MParams')



#run model calibration for quantization
with torch.no_grad():
    stats = collect_stats(model, calib_data, num_batches=2, device=device)
    amax = compute_amax(model, method="percentile", percentile=99.99, device=device)
    
    # optional - test different calibration methods
    #amax = compute_amax(model, method="mse")
    #amax = compute_amax(model, method="entropy")



#run model evaluation
# set desired approximate multiplier in each layer
#at first, set all layers to have the 8-bit accurate multiplier
axx_list = [{'axx_mult' : 'mul8s_acc', 'axx_power' : 1.0, 'quant_bits' : 8, 'fake_quant' : False}]*len(conv2d_layers)

# For example, set the first 10 layers to be approximated with a specific multiplier 
axx_list[0:len(conv2d_layers)] = [{'axx_mult' : 'bw_mult_9_9_0', 'axx_power' : 0.7082, 'quant_bits' : 9, 'fake_quant' : False}] * len(conv2d_layers)

returned_power = [0]
replace_conv_layers(model,  AdaptConv2D, axx_list, total_macs, total_params, layer_count=[0], returned_power = returned_power, initialize = False)  
print(model)
print('Power of approximated operations: ', round(returned_power[0], 2), '%')
print('Model compiled. Running evaluation')

# Run evaluation on the validation dataset
top1 = evaluate_cifar10(model, val_data, device = device)



#run model retraining
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001) # set desired learning rate
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

#one epoch retrain
train_one_epoch(model, criterion, optimizer, calib_data, device, 0, 10)



#re-run model evaluation
top1 = evaluate_cifar10(model, val_data, device = device)