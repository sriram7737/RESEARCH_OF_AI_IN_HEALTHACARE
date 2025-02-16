import torch
print("PyTorch Version:", torch.__version__)          # Should print something like '2.0.1+cu118'
print("CUDA Available:", torch.cuda.is_available())    # Should print 'True'
print("CUDA Version:", torch.version.cuda)             # Should print '11.8'
print("GPU Count:", torch.cuda.device_count())         # Should print '1' or the number of GPUs
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))  # Should print your GPU's name
