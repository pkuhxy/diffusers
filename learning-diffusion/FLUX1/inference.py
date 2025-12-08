import os
import torch
from diffusers import FluxPipeline

# Load the model and set the torch dtype to bfloat16 for efficiency
pipe = FluxPipeline.from_pretrained("/apdcephfs_nj7/share_1220751/xianyihe/ckpts/black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()  # Save VRAM by offloading the model to CPU

# Define the path to your txt file containing prompts
prompt_file_path = "prompts.txt"  # Adjust to the correct path for your .txt file

# Read prompts from the file
with open(prompt_file_path, 'r') as file:
    prompts = [line.strip() for line in file.readlines()]

# Directory to store generated images
output_dir = "geneval"  # Main output directory

# Make sure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Loop through each prompt and generate an image
for idx, prompt in enumerate(prompts, 1):
    # Create a subfolder for each image, named 1, 2, 3, ...
    save_dir = os.path.join(output_dir, str(idx))
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate the image
    image = pipe(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0),
        show_trajectory=True,
        save_dir=save_dir,  # Save the image to the subfolder
    ).images[0]
    
    # Save the image to the respective subfolder
    image_path = os.path.join(save_dir, f"image_{idx}.png")
    image.save(image_path)
    print(f"Image saved at: {image_path}")
