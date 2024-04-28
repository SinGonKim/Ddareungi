import numpy as np
from sklearn.metrics import accuracy_score as accuracy
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# If you want to use custom metrics, please add functions.

def cal_log(log_dict, **kwargs):
    for key, value in log_dict.items():
        try:
            value.append(globals()[key.split("_")[-1]](kwargs))
        except RuntimeError:
            print(kwargs['loss'])
            print(kwargs['loss'].data)
    return log_dict


def loss(kwargs):
    return float(kwargs['loss'].data.cpu().numpy()) # Unable to get repr for <class 'torch.Tensor'> , {RuntimeError}CUDA error: device-side assert triggered Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.



def acc(kwargs):
    return accuracy(np.argmax(kwargs['outputs'].data.cpu().numpy(), axis=1), kwargs['labels'].data.cpu().numpy())
