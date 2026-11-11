import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def generate_grad_cam(model, img_tensor, target_layer, target_class=None):
    """
    Generates a Grad-CAM heatmap for a given image and model.

    Args:
        model (torch.nn.Module): The pre-trained CNN model.
        img_tensor (torch.Tensor): The preprocessed input image tensor.
        target_layer (torch.nn.Module): The target convolutional layer for Grad-CAM.
        target_class (int, optional): The index of the target class. If None,
                                      the class with the highest predicted probability is used.

    Returns:
        numpy.ndarray: The Grad-CAM heatmap.
    """
    model.eval()

    # Store gradients and activations
    activations = []
    gradients = []

    def save_activations(module, input, output):
        activations.append(output)

    def save_gradients(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # Register hooks to capture activations and gradients
    hook_handle_activations = target_layer.register_forward_hook(save_activations)
    hook_handle_gradients = target_layer.register_full_backward_hook(save_gradients)

    # Forward pass
    output = model(img_tensor)

    # Get target class if not provided
    if target_class is None:
        target_class = output.argmax(dim=1).item()

    # Zero all gradients
    model.zero_grad()

    # Backward pass for the target class
    one_hot_output = torch.zeros_like(output)
    one_hot_output[0][target_class] = 1
    output.backward(gradient=one_hot_output, retain_graph=True)

    # Remove hooks
    hook_handle_activations.remove()
    hook_handle_gradients.remove()

    # Get features and gradients
    feature_map = activations[0].squeeze(0)
    grads_val = gradients[0].squeeze(0)

    # Global Average Pooling of gradients
    weights = F.adaptive_avg_pool2d(grads_val, 1)

    # Weighted combination of feature maps
    cam = (weights * feature_map).sum(dim=0)
    cam = F.relu(cam)  # Apply ReLU

    # Normalize heatmap
    cam = cam - cam.min()
    cam = cam / cam.max()

    return cam.detach().cpu().numpy()

# Example Usage:
if __name__ == "__main__":
    # Load a pre-trained ResNet model and the names of its 1,000 ImageNet classes
    weights = models.ResNet50_Weights.IMAGENET1K_V1
    model = models.resnet50(weights=weights)
    model.eval()
    categories = weights.meta["categories"]

    # Define image transformations
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load and preprocess an image
    here = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(here, "images", "cat.jpg")  # Replace with your image path
    original_image = Image.open(img_path).convert('RGB')
    img_tensor = preprocess(original_image).unsqueeze(0)

    # Define the target layer (e.g., the last convolutional layer)
    target_layer = model.layer4[-1] # For ResNet50, this is a good choice

    # Print the model's top predictions, so we know which classes the heatmaps explain
    with torch.no_grad():
        probs = model(img_tensor).softmax(dim=1)[0]
    top = probs.topk(5)
    print("Top 5 predictions:")
    for p, idx in zip(top.values.tolist(), top.indices.tolist()):
        print(f"  {categories[idx]:<15} {p:6.1%}   (class {idx})")

    # Explain the top two predictions side by side.
    # To explain a class of your choice instead, use e.g. [categories.index("tabby")]
    target_classes = top.indices[:2].tolist()

    # Draw the original image, then one Grad-CAM overlay per class
    extent = (0, original_image.width, original_image.height, 0)  # stretches the 7x7 map over the photo
    fig, axes = plt.subplots(1, 1 + len(target_classes), figsize=(5 * (1 + len(target_classes)), 3.6))
    axes[0].imshow(original_image)
    axes[0].set_title("Original image")
    for ax, target_class in zip(axes[1:], target_classes):
        heatmap = generate_grad_cam(model, img_tensor, target_layer, target_class)
        ax.imshow(original_image)
        overlay = ax.imshow(heatmap, cmap='jet', alpha=0.5, extent=extent, vmin=0, vmax=1)
        ax.set_title(f"Grad-CAM for \"{categories[target_class]}\" ({probs[target_class]:.1%})")
    for ax in axes:
        ax.axis('off')

    # One colour bar for all heatmaps. Each heatmap is scaled to 0-1 on its own,
    # so red means "this image's strongest support for that class", not a probability.
    colour_bar = fig.colorbar(overlay, ax=axes, fraction=0.025, pad=0.02, shrink=0.8)
    colour_bar.set_label("Relative support (0 = least, 1 = most)")

    out_path = os.path.join(here, "heatmap.jpg")
    fig.savefig(out_path, bbox_inches="tight")  # save before show, or the file can be blank
    print(f"Saved {out_path}")
    plt.show()