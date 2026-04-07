import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the transformer model
model = HookedTransformer.from_pretrained("gpt2-small").to(device)

# Load the sparse autoencoder
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.8.hook_resid_pre",
    device=device
)

# Define input text
text = "Let's activate some neurons to understand what the model is doing!"

# Tokenize input
tokens = model.to_tokens(text).to(device)

# Create a list to store the activations
stored_activations = []

# Define a hook function that stores the activations
def store_hook(act, hook):
    stored_activations.append(act.detach())
    return act  # Return unchanged to not affect the forward pass

# Pass through GPT-2 and extract activations
with torch.no_grad():
    model.run_with_hooks(tokens, fwd_hooks=[("blocks.8.hook_resid_pre", store_hook)])

# Get the stored activations
activations = stored_activations[0]

# Now encode the activations
encoded_activations = sae.encode(activations)

top_k = 20  # Number of top features to display
activation_sums = encoded_activations.sum(dim=(0, 1)).cpu()  # Sum over batch and sequence dimensions
top_indices = torch.topk(activation_sums, top_k).indices.tolist()

print(f"Top {top_k} activated features:")
for i, idx in enumerate(top_indices, 1):
    print(f"{i}. Feature {idx}: {activation_sums[idx].item():.4f}")

# Convert to numpy for visualization
act_np = encoded_activations.detach().squeeze().cpu().numpy()  # Remove batch dimension

# Plot the sparsity pattern
plt.figure(figsize=(12, 8))
plt.imshow(act_np > 0, cmap='binary', aspect='auto')
plt.colorbar(label='Active (1) / Inactive (0)')
plt.xlabel('Feature Index')
plt.ylabel('Token Position')
plt.title('Sparsity Pattern of SAE Activations')
plt.tight_layout()
plt.show()

plt.savefig('mech.png')

# Calculate sparsity statistics
sparsity_ratio = (act_np == 0).mean() * 100
print(f"Sparsity: {sparsity_ratio:.2f}% of activations are zero")