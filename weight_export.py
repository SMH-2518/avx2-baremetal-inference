import torch
import torch.nn as nn
import struct

# 1. Define your PyTorch Model
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Example sizes: 784 (MNIST) -> 128 -> 10
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleMLP()

# 2. Extract and Pack into Raw Binary
def export_to_bin(model, filename="weights.bin"):
    with open(filename, "wb") as f:
        layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
        
        # Write number of layers
        f.write(struct.pack("i", len(layers)))
        
        for layer in layers:
            in_features = layer.in_features
            out_features = layer.out_features
            
            # Write dimensions
            f.write(struct.pack("i", in_features))
            f.write(struct.pack("i", out_features))
            
            # PyTorch stores weights as [out_features, in_features] (Transposed)
            # Flatten to 1D and write as standard floats
            weights = layer.weight.detach().numpy().flatten()
            f.write(struct.pack(f"{len(weights)}f", *weights))
            
            # Write bias
            bias = layer.bias.detach().numpy().flatten()
            f.write(struct.pack(f"{len(bias)}f", *bias))
            
    print(f"Exported {len(layers)} layers to {filename}")

export_to_bin(model)