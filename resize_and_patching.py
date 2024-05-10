import torch
import torch.nn.functional as F

# Given tensor
input_tensor = torch.randn(1000, 1000, 3)

# Desired width (w) and height (h)
desired_w, desired_h = 500, 500

# Resizing the tensor
# First, permute the tensor to have channels in the first dimension (C, H, W)
input_tensor_permuted = input_tensor.permute(2, 0, 1).unsqueeze(0)

# Resize using interpolate
resized_tensor = F.interpolate(input_tensor_permuted, size=(desired_h, desired_w), mode='bilinear', align_corners=False)

# Permute back to original dimensions (H, W, C)
output_tensor = resized_tensor.squeeze(0).permute(1, 2, 0)

print(output_tensor.shape)


import numpy as np

def split_image_into_patches(image, patch_size=1000):
    
    # Check if the image dimensions are divisible by the patch size
    assert image.shape[0] % patch_size == 0, "Image height must be divisible by the patch size"
    assert image.shape[1] % patch_size == 0, "Image width must be divisible by the patch size"

    # Calculate the number of patches along each dimension
    m = image.shape[0] // patch_size
    n = image.shape[1] // patch_size

    patches = []

    # Split the image
    for i in range(m):
        for j in range(n):
            # The slice includes all channels
            patch = image[i * patch_size:(i + 1) * patch_size,
                          j * patch_size:(j + 1) * patch_size, :]
            patches.append(patch)

    patches = np.stack(patches, axis =1)
    return patches

# Example usage:
large_image = np.random.rand(1000, 1000, 3)  # Example large color image
patches = split_image_into_patches(large_image, patch_size=100)

print(patches.shape)
