import os
import json

def generate_image_json(root_folder, output_json_file):
    # 支持的图像格式，可根据需要添加
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    
    image_data = {}
    index = 1
    
    # 遍历文件夹
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            # 获取文件后缀名（转换为小写比较）
            ext = os.path.splitext(filename)[1].lower()
            
            if ext in image_extensions:
                # 获取文件的绝对路径
                full_path = os.path.join(dirpath, filename)
                # 获取相对于根目录的相对路径
                relative_path = os.path.relpath(full_path, root_folder)
                
                # 按照要求存入字典
                image_data[index] = {
                    "path": relative_path
                }
                index += 1
    
    # 将字典保存到 JSON 文件
    try:
        with open(output_json_file, 'w', encoding='utf-8') as f:
            json.dump(image_data, f, indent=4, ensure_ascii=False)
        print(f"✅ 成功! 已扫描 {index-1} 张图片，结果保存在: {output_json_file}")
    except Exception as e:
        print(f"❌ 保存 JSON 文件时出错: {e}")

# --- 配置 ---
# 在这里修改你的文件夹路径
target_folder = r"/apdcephfs_nj7/share_1220751/xianyihe/dataset/BestWishYsh/OpenS2V-Eval/Images/humanobj"  
output_file = "opens2v.json"

# --- 执行 ---
if __name__ == "__main__":
    # 简单的路径检查
    if os.path.exists(target_folder):
        generate_image_json(target_folder, output_file)
    else:
        print(f"❌ 找不到文件夹: {target_folder}")
