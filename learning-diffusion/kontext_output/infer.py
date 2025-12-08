import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image

pipe = FluxKontextPipeline.from_pretrained("/apdcephfs_nj7/share_1220751/xianyihe/ckpts/black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
pipe.to("cuda")

input_image = load_image("/apdcephfs_nj7/share_1220751/xianyihe/dataset/BestWishYsh/OpenS2V-Eval/Images/humanobj/human/crop_woman/celebrity/2.jpg")

image = pipe(
  image=input_image,
  prompt="A woman is ice skating",
  guidance_scale=2.5
).images[0]
image.save("ski.png")
