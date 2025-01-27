import torch

def get_device():
    """
    Determine the available device for computation.
    Returns either CUDA device if available, or CPU.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu') 