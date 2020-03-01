import os,sys
import torch

data_root = os.path.abspath('../../inputs/fashion-mnist')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float16
fig_size = 28
model_saved_path = os.path.abspath('./model')

log_dir = os.path.abspath('.logs')
if os.path.exists(log_dir) is not True:
    os.makedirs(log_dir)

val_len = 5000


# block_arch = [(64, 64, 2, True), (64, 128, 2), (128, 256, 2), (256, 512, 2)]
# None : 
#       valid acc:  0.9275833333333333 test acc:  0.9272
# dropout : 
#       valid acc:  0.9259166666666667 test acc:  0.9211
# smaller learning rate:
#       valid acc:  0.9239166666666667 test acc:  0.9095
# cos lr shceduler:
#       valid acc:  valid acc:  0.9316666666666666 test acc:  0.9356
# data augmentation:
#       
# different optimizer:
#       SGD:
#       Adam:
#       Adabound: