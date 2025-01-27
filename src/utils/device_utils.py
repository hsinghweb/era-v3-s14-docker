import torch
import os

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return 'cpu'

def set_seed(seed=1337):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def save_model(model, optimizer, loss, epoch, path='model.pt'):
    # Convert model to half precision
    model_to_save = model.half()
    
    # Save in half precision
    torch.save({
        'model_state_dict': model_to_save.state_dict(),
        'loss': loss,
        'epoch': epoch
    }, path, _use_new_zipfile_serialization=False)
    
    # Convert back to original precision
    model.float()
    print(f"Model saved to {path}")

def load_model(model, optimizer=None, path='model.pt'):
    if os.path.exists(path):
        checkpoint = torch.load(path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint.get('epoch', 0), checkpoint.get('loss', None)
    return 0, None 