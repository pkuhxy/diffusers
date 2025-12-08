import os
import json
import torch
import torch.nn as nn
import torchvision.transforms as T
from transformers import AutoImageProcessor, CLIPProcessor, CLIPModel
from PIL import Image
from tqdm import tqdm  # 补全缺失的库

# --- 1. 补充缺失的 Transform 函数 ---
def get_size(size):
    if isinstance(size, int):
        return (size, size)
    elif "height" in size and "width" in size:
        return (size["height"], size["width"])
    elif "shortest_edge" in size:
        return size["shortest_edge"]
    else:
        raise ValueError(f"Invalid size: {size}")

def get_image_transform(processor: AutoImageProcessor):
    config = processor.to_dict()
    # 注意：ToTensor 应该放在最前面，但这里的逻辑是接收 Tensor 或 PIL
    # 为了通用性，我们在 T.Compose 里不包含 ToTensor，而是假设输入已经是 Tensor 或者在外部转
    # 修正逻辑：HuggingFace 的处理器通常期望输入是 Tensor(0-1) 或者 PIL
    
    transforms_list = []
    
    # 1. Resize
    if config.get("do_resize"):
        transforms_list.append(T.Resize(get_size(config.get("size")), interpolation=T.InterpolationMode.BICUBIC))
    
    # 2. Crop
    if config.get("do_center_crop"):
        transforms_list.append(T.CenterCrop(get_size(config.get("crop_size"))))
        
    # 3. ToTensor (确保是 Tensor 且是 0-1 之间，这是 Normalize 的前提)
    # 如果输入已经是 Tensor，这步通常会被忽略，但为了保险起见，建议在 transform 外部控制 ToTensor
    
    # 4. Normalize
    if config.get("do_normalize"):
        transforms_list.append(T.Normalize(mean=processor.image_mean, std=processor.image_std))

    return T.Compose(transforms_list)

# --- 2. 你的 ClipScorer 类 (修正版) ---
class ClipScorer(torch.nn.Module):
    def __init__(self, device, model_path):
        super().__init__()
        self.device = device
        # 将路径参数化
        print(f"Loading CLIP model from {model_path}...")
        self.model = CLIPModel.from_pretrained(model_path).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_path)
        
        # 获取预处理管线
        # 注意：我们需要一个能直接处理 PIL 图片转为最终 Tensor 的管线
        self.image_processor = self.processor.image_processor
        self.tform = get_image_transform(self.image_processor)
        
        self.eval()

    def process_image(self, image: Image.Image):
        """
        处理单张 PIL 图片：转 Tensor -> Resize -> Crop -> Normalize
        返回: (C, H, W) 的 Tensor
        """
        # 1. 确保转为 Tensor (0-1)
        tensor_img = T.ToTensor()(image) 
        # 2. 应用 Resize/Crop/Normalize
        # 注意：T.Resize 等在处理 Tensor 时需要 (C, H, W)
        processed_img = self.tform(tensor_img)
        return processed_img

    @torch.no_grad()
    def __call__(self, pixels, prompts, return_img_embedding=False):
        """
        pixels: 已经是预处理好的 Batch Tensor (B, C, H, W)
        prompts: 文本列表
        """
        # 文本处理
        texts = self.processor(text=prompts, padding='max_length', truncation=True, return_tensors="pt").to(self.device)
        
        # 图像只需要移动到设备，不需要再 _process，因为 main 中已经处理好了
        pixels = pixels.to(self.device)
        
        outputs = self.model(pixel_values=pixels, **texts)
        
        if return_img_embedding:
            # 注意：diagonal 用于计算一一对应的相似度
            return outputs.logits_per_image.diagonal() / 30, outputs.image_embeds
        return outputs.logits_per_image.diagonal() / 30

# --- 3. 主执行逻辑 ---
def main():
    # === 配置路径 ===
    PROMPTS_FILE = "prompts.txt"
    IMAGE_ROOT_DIR = "geneval"
    OUTPUT_JSON = "similarity_scores_2.json"
    BATCH_SIZE = 24
    # 建议修改为你实际的模型路径
    MODEL_PATH = "/apdcephfs_nj7/share_1220751/xianyihe/ckpts/openai/clip-vit-large-patch14" 
    
    # 检查路径是否存在，不存在则使用默认的 openai 路径（方便调试）
    if not os.path.exists(MODEL_PATH):
        print(f"Path {MODEL_PATH} not found, using 'openai/clip-vit-large-patch14' from Hub.")
        MODEL_PATH = "openai/clip-vit-large-patch14"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = ClipScorer(device, model_path=MODEL_PATH)
    
    # === 读取 Prompts ===
    if not os.path.exists(PROMPTS_FILE):
        print(f"Error: {PROMPTS_FILE} not found.")
        return

    print(f"Reading prompts from {PROMPTS_FILE}...")
    prompt_map = {}
    with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            folder_name = str(idx + 1)
            prompt_map[folder_name] = line.strip()

    results = {}

    # === 遍历文件夹 ===
    if not os.path.exists(IMAGE_ROOT_DIR):
        print(f"Error: Directory {IMAGE_ROOT_DIR} not found.")
        return

    subfolders = [d for d in os.listdir(IMAGE_ROOT_DIR) if os.path.isdir(os.path.join(IMAGE_ROOT_DIR, d))]
    subfolders.sort(key=lambda x: int(x) if x.isdigit() else float('inf'))

    for folder_name in tqdm(subfolders, desc="Processing folders"):
        if folder_name not in prompt_map:
            # 这里的警告可能太多，视情况开启
            # print(f"Warning: Folder '{folder_name}' not in prompts map.")
            continue
            
        current_prompt = prompt_map[folder_name]
        folder_path = os.path.join(IMAGE_ROOT_DIR, folder_name)
        
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        if not image_files:
            continue

        folder_results = {}
        
        # === 批处理图像 ===
        for i in range(0, len(image_files), BATCH_SIZE):
            batch_files = image_files[i : i + BATCH_SIZE]
            batch_tensors = []
            valid_files = []
            
            # 加载并预处理图片
            for img_file in batch_files:
                try:
                    img_path = os.path.join(folder_path, img_file)
                    img = Image.open(img_path).convert("RGB")
                    
                    # 关键修改：在 stack 之前，调用 scorer.process_image 进行 Resize/Crop/Norm
                    # 这样出来的 tensor 都是 (3, 224, 224)，可以被 stack
                    processed_tensor = scorer.process_image(img)
                    batch_tensors.append(processed_tensor)
                    valid_files.append(img_file)
                except Exception as e:
                    print(f"Error loading {img_file}: {e}")
            
            if not batch_tensors:
                continue

            # 堆叠成 (Batch, 3, H, W)
            # 现在这一步是安全的，因为所有 tensor 尺寸已被 scorer.process_image 统一
            pixel_batch = torch.stack(batch_tensors)

            # 构造对应数量的 prompt
            prompts_batch = [current_prompt] * len(batch_tensors)
            
            # 计算分数
            try:
                scores = scorer(pixel_batch, prompts_batch)
                scores = scores.cpu().tolist()
                
                for fname, score in zip(valid_files, scores):
                    folder_results[fname] = score
            except RuntimeError as e:
                print(f"Runtime Error in folder {folder_name}: {e}")
                torch.cuda.empty_cache()

        results[folder_name] = {
            "prompt": current_prompt,
            "scores": folder_results
        }

    # === 保存结果 ===
    print(f"Saving results to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print("Done!")

if __name__ == "__main__":
    main()
