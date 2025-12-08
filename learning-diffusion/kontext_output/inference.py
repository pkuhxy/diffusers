import torch
import json
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
import os
import argparse

# 初始化模型
pipe = FluxKontextPipeline.from_pretrained("/apdcephfs_nj7/share_1220751/xianyihe/ckpts/black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
pipe.to("cuda")

# 配置路径
json_path = "/apdcephfs_nj7/share_1220751/xianyihe/dataset/sysuyy/ImgEdit/Benchmark/singleturn/singleturn.json"  # 替换为实际的JSON文件路径
root_image_path = "/apdcephfs_nj7/share_1220751/xianyihe/dataset/sysuyy/ImgEdit/Benchmark/singleturn"  # 替换为图像根路径
output_dir = "trajecotory_denoise_SDE"

os.makedirs(output_dir, exist_ok=True)

# 设置 argparse
parser = argparse.ArgumentParser(description='处理图像的部分数据')
parser.add_argument('--part', type=int, required=True, help='要处理的部分编号 (0 到 total_part-1)')
args = parser.parse_args()

part = args.part
total_part = 8

# 读取JSON文件
with open(json_path, 'r') as f:
    data = json.load(f)

data_list = [(key, item) for key, item in data.items()]

length = len(data_list)

chunk_length = length // total_part

# 检查 part 是否在有效范围内
if part < 0 or part >= total_part:
    raise ValueError(f"part 必须在 0 到 {total_part-1} 之间，当前为 {part}")

for i in range(chunk_length * part, chunk_length * (part + 1)):

    key, item = data_list[i]

    # 检查edit_type是否符合条件
    # if item.get("edit_type") in ["extract", "action"]:
        # 拼接完整图像路径
    image_path = os.path.join(root_image_path, item["id"])
        # 为每个 key 创建独立的输出目录
    
    
    current_output_dir = os.path.join(output_dir, key)
    os.makedirs(current_output_dir, exist_ok=True)
    

    for i in range(5):

        current_output_dir_SDE = os.path.join(current_output_dir,str(i))
        os.makedirs(current_output_dir_SDE, exist_ok=True)
        
        print(i)

        try:
            # 加载图像
            input_image = load_image(image_path)
            
            # 使用模型处理图像
            image = pipe(
                image=input_image,
                prompt=item["prompt"],
                guidance_scale=2.5,
                show_trajectory=True,
                save_dir=current_output_dir_SDE,
            ).images[0]
            
            # 保存结果（可根据需要修改保存路径和命名规则）
            output_path = f"output_{key}.png"
            # image.save(output_path)
            print(f"处理成功：{key}，结果保存至：{output_path}")
            
        except Exception as e:
            print(f"处理失败：{key}，错误信息：{str(e)}")

print(f"部分 {part} 的所有符合条件的图像处理完成！")
