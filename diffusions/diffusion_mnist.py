import time
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from diffusers import UNet2DModel, DDPMScheduler

# -------------------- Create Checkpoints Folder --------------------
checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)  # Create the folder if it doesn't exist

# -------------------- Hyperparameters --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 128
num_train_timesteps = 1000            # Total timesteps for the diffusion process
lr = 1e-4           # Learning rate for the optimizer
num_epochs = 100        # Total training epochs
image_size = 32     # Image resolution (MNIST images are 28x28)

# -------------------- Dataset & Dataloader --------------------
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),  # Converts images to PyTorch tensors and scales pixel values to [0, 1]
    transforms.Normalize((0.5,), (0.5,)) # Normalizes to [-1, 1] via: (x - 0.5) / 0.5
])
# Set download=True if the dataset is not already available in the specified directory.
dataset = datasets.MNIST('.', train=True, download=False, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

# -------------------- Model & Scheduler --------------------
# Using diffusers' UNet2DModel for a 1-channel image (MNIST)
model = UNet2DModel(
    sample_size=image_size,
    in_channels=1,
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(64, 128, 256, 512),
    down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
).to(device)

# Define a scheduler – this handles the noise schedule and denoising steps
scheduler = DDPMScheduler(
    num_train_timesteps=num_train_timesteps,
    beta_start=0.0001,
    beta_end=0.02,
    beta_schedule="linear",
)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# # Setup for AMP
# scaler = torch.amp.GradScaler('cuda')

# -------------------- Log File Setup --------------------
log_filename = os.path.join(checkpoint_dir, 'training_log.txt')
# Open the log file in write mode (this clears any previous logs)
with open(log_filename, 'w') as log_file:
    log_file.write("Training Log\n")
    log_file.write("=" * 50 + "\n")

# -------------------- Training Loop --------------------
model.train()   # Set model to training mode
for epoch in range(num_epochs):
    # Record the start time of the epoch
    epoch_start_time = time.time()
    running_loss = 0.0  # Track cumulative loss for the epoch
    
    for images, _ in dataloader:
        images = images.to(device)
        batch_size_current = images.shape[0]
        
        # Randomly sample a timestep for each image in the batch (shape: [batch_size])
        timesteps = torch.randint(0, num_train_timesteps, (batch_size_current,), device=device).long()
        
        # Generate random noise matching the image shape
        noise = torch.randn_like(images)
        
        # Use the scheduler to add noise to the images according to the DDPM process
        noisy_images = scheduler.add_noise(images, noise, timesteps)
        
        # optimizer.zero_grad()
        
        # # Mixed precision forward and backward pass
        # with torch.amp.autocast('cuda'):
        #     noise_pred = model(noisy_images, timesteps).sample
        #     loss = nn.MSELoss()(noise_pred, noise)
        
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        
        # Use the model to predict the noise component from the noisy images.
        # Note: The model returns a UNet2DModelOutput; we use the .sample attribute.
        noise_pred = model(noisy_images, timesteps).sample
        
        # Calculate MSE loss between the predicted noise and the actual noise.
        loss = nn.MSELoss()(noise_pred, noise)
        
        # Backpropagation: Clear gradients, compute new gradients, and update weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate loss, scaled by the batch size.
        running_loss += loss.item() * batch_size_current
    
    # Compute the average loss for the epoch
    epoch_loss = running_loss / len(dataloader.dataset)
    
    # Calculate the running time for the epoch
    epoch_time = time.time() - epoch_start_time
    
    # Create a log message with epoch, loss, and running time information
    log_message = f"Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss:.4f} | Time: {epoch_time:.2f} sec"
    
    # Print the log message to the console
    print(log_message)
    
    # Append the log message to the text file
    with open(log_filename, 'a') as log_file:
        log_file.write(log_message + "\n")
    
    # Save the model checkpoint every 20 epochs
    if (epoch + 1) % 20 == 0:
        save_path = os.path.join(checkpoint_dir, f"diffusion_mnist_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)
        save_message = f"Model saved to {save_path}"
        print(save_message)
        with open(log_filename, 'a') as log_file:
            log_file.write(save_message + "\n")