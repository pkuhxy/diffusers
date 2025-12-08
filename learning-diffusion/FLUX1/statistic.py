import json
import os
import re
import matplotlib.pyplot as plt

# 假设你的json文件名为 data.json，如果在代码中直接定义数据，可以略过读取文件部分
# 这是一个模拟的输入数据，用于演示（包含了你提供的样例）

def plot_scores(data, output_folder="score_charts"):
    # 1. 创建保存文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"创建文件夹: {output_folder}")

    # 2. 遍历字典
    for key, content in data.items():
        scores_dict = content.get("scores", {})
        prompt = content.get("prompt", "")
        
        # 用于存储 (step, score) 的列表
        points = []

        # 3. 提取数据
        for filename, score in scores_dict.items():
            # 使用正则匹配 step_XXX_grid.png 格式
            match = re.search(r'step_(\d+)_grid\.png', filename)
            
            if match:
                # 提取数字部分并转为 int
                step_num = int(match.group(1))
                points.append((step_num, score))
            else:
                # 过滤掉不符合格式的文件，比如 "image_3.png"
                pass

        # 4. 排序数据 (按 step 从小到大)
        if not points:
            print(f"Key {key} 没有有效数据，跳过。")
            continue
            
        points.sort(key=lambda x: x[0])

        # 分离 x 和 y 轴数据
        x = [p[0] for p in points]
        y = [p[1] for p in points]

        # 5. 绘图
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, marker='o', linestyle='-', markersize=4, label='Score')
        
        # 图表装饰
        plt.title(f"Score Trend for ID: {key}\nPrompt: {prompt}", fontsize=12)
        plt.xlabel("Step Number")
        plt.ylabel("Score Value")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 6. 保存图片
        save_path = os.path.join(output_folder, f"{key}.png")
        plt.savefig(save_path)
        plt.close() # 关闭画布，防止内存溢出
        
        print(f"已保存图表: {save_path}")

# --- 主程序 ---
if __name__ == "__main__":
    # 如果你是从文件读取，请取消下面两行的注释，并注释掉 json.loads(json_data_str)
    with open('similarity_scores_2.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 加载数据
    # data = json.loads(json_data_str)
    
    # 执行绘图函数
    plot_scores(data, output_folder="output_charts_2")
