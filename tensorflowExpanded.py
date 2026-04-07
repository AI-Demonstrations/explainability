import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import math

def main():
    # Initialize the writer in a new folder
    writer = SummaryWriter('runs/tensorboard_showcase')
    print("Generating a massive TensorBoard showcase...")

    # --- 1. MULTIPLE SCALARS (Comparing lines on one graph) ---
    for step in range(100):
        # Simulating a Training vs. Validation Loss scenario
        train_loss = math.exp(-step * 0.05) + (torch.rand(1).item() * 0.1)
        val_loss = math.exp(-step * 0.04) + (torch.rand(1).item() * 0.2)
        
        # add_scalars (plural) puts multiple lines on the exact same chart!
        # Note: This creates a separate folder structure under the hood.
        writer.add_scalars('Loss/Comparison', {
            'Training': train_loss,
            'Validation': val_loss
        }, step)

    # --- 2. IMAGES (Viewing a grid of pictures) ---
    # Create a batch of 16 random "color images" (Batch Size, Channels, Height, Width)
    # We will make them look a bit like colorful TV static
    dummy_images = torch.rand(16, 3, 64, 64) 
    
    # torchvision's make_grid is the standard way to display a batch of images neatly
    img_grid = torchvision.utils.make_grid(dummy_images, nrow=4)
    writer.add_image('Visuals/Random_Image_Batch', img_grid, global_step=0)

    # --- 3. HISTOGRAMS (Tracking data distributions over time) ---
    for step in range(50):
        # Simulating neural network weights that start at zero, but shift upward over time
        fake_weights = torch.randn(1000) + (step * 0.1) 
        writer.add_histogram('Model_Internals/Layer_1_Weights', fake_weights, step)

    # --- 4. MATPLOTLIB FIGURES (Custom charts) ---
    # You can draw ANY custom chart using standard Matplotlib and send it directly to TensorBoard
    fig, ax = plt.subplots()
    x_vals = np.linspace(0, 10, 100)
    y_vals = np.sin(x_vals) * np.exp(-x_vals * 0.1) # A damped sine wave
    ax.plot(x_vals, y_vals, color='purple', linewidth=2)
    ax.set_title("Damped Sine Wave (Custom Matplotlib Chart)")
    
    # Notice we use add_figure instead of add_image here
    writer.add_figure('Custom_Charts/Matplotlib_Integration', fig, global_step=0)

    # --- 5. TEXT (Logging notes or configurations) ---
    # TensorBoard text fully supports Markdown formatting!
    experiment_details = """
    ### Experiment Run: Alpha-Zero
    * **Learning Rate:** 0.001
    * **Batch Size:** 16
    * **Optimizer:** Adam
    
    *Note: This run uses the simulated dampening effect on the validation curve.*
    """
    writer.add_text('Experiment_Config/Notes', experiment_details, global_step=0)

    # Always close the writer
    writer.close()
    print("Done! You can now start the TensorBoard server.")

if __name__ == "__main__":
    main()