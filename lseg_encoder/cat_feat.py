import torch
import os

# Path where your tensors are located
tensor_directory = "/public/home/renhui/code/4d/feature-4dgs/data/train_480/code_output/rgb_feature_langseg"

# Initialize an empty list to store tensors
tensor_list = []

# Loop through the files in the range
for i in range(0, 80):  # From 000001 to 000079 (inclusive)
    filename = f"{i:05d}_fmap_CxHxW.pt"
    file_path = os.path.join(tensor_directory, filename)

    # Load the tensor and append it to the list
    try:
        tensor = torch.load(file_path)
        tensor_list.append(tensor.unsqueeze(0))
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")

# Concatenate all tensors along a new dimension (dim=0, you can change it based on your needs)
combined_tensor = torch.cat(tensor_list, dim=0)

print('combine tensor', combined_tensor.shape)
# Save the combined tensor
output_path = os.path.join(tensor_directory, "langseg_feats.pth")
torch.save(combined_tensor, output_path)

print(f"Combined tensor saved at {output_path}")