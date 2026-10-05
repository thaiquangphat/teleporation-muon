import os
import torch
import random
import numpy as np 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)              # NumPy RNG
    torch.manual_seed(seed)           # PyTorch CPU RNG
    torch.cuda.manual_seed(seed)      # PyTorch CUDA RNG for single GPU
    torch.cuda.manual_seed_all(seed)  # PyTorch CUDA RNG for multi-GPU
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(save_path, model, optimizer, scheduler, epoch, val_acc, lr):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'val_acc': val_acc,
        'learning_rate': lr,
    }, save_path)
def remove_old_checkpoints(directory: str, prefix: str):
    """
    Remove all files in the given directory that start with the given prefix.
    """
    for fname in os.listdir(directory):
        if fname.startswith(prefix):
            path = os.path.join(directory, fname)
            if os.path.isfile(path):
                os.remove(path)
