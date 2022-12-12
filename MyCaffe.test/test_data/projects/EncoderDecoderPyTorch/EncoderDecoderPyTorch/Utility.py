import torch

def free_memory(item):
    item.cpu()
    del item

def report_memory(device):
    if device.type == 'cuda':
        print(torch.cuda.memory_summary(device=device, abbreviated=False))
    
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r-a  # free inside reserved
        print(f'total: {t}, reserved: {r}, allocated: {a}, free: {f}')
    