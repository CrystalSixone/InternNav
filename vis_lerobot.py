"""
可视化 LeRobot 数据集中的 episode

用法:
    python vis_lerobot.py --dataset_path <lerobot数据集路径> --key <scan_episode_id> [--output_dir <输出目录>] [--fps <帧率>]
    
示例:
    python vis_lerobot.py --dataset_path data/vln_pe/traj_data/gruvln10 --key scan_05_0
    python vis_lerobot.py --dataset_path data/vln_pe/traj_data/gruvln10 --key scan_05_0 --output_dir vis_output --fps 5
"""

import os
import argparse
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import json


def get_font(font_size):
    """
    获取字体，优先尝试支持中文的字体
    """
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]
    try:
        for path in font_paths:
            if os.path.exists(path):
                return ImageFont.truetype(path, font_size)
        return ImageFont.load_default()
    except:
        return ImageFont.load_default()


def wrap_text(text, font, max_width):
    """
    将文本换行以适应指定宽度
    """
    lines = []
    words = text.split(' ')
    current_line = ""
    
    for word in words:
        test_line = current_line + " " + word if current_line else word
        bbox = font.getbbox(test_line)
        text_width = bbox[2] - bbox[0]
        
        if text_width <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    
    if current_line:
        lines.append(current_line)
    
    return lines


def calculate_text_height(text, font, max_width, line_spacing=5):
    """
    计算文本在给定宽度下换行后的总高度
    """
    wrapped_lines = wrap_text(text, font, max_width)
    bbox = font.getbbox("Ag")  # 使用典型字符计算行高
    line_height = (bbox[3] - bbox[1]) + line_spacing
    return len(wrapped_lines) * line_height, wrapped_lines


def find_optimal_font_size(text, panel_width, panel_height, margin=10, title_height=35, min_font_size=10, max_font_size=24):
    """
    找到能让文本完全显示的最大字体大小
    """
    available_width = panel_width - 2 * margin
    available_height = panel_height - title_height - 2 * margin
    
    # 从大到小尝试字体大小
    for font_size in range(max_font_size, min_font_size - 1, -1):
        font = get_font(font_size)
        text_height, wrapped_lines = calculate_text_height(text, font, available_width, line_spacing=5)
        
        if text_height <= available_height:
            return font_size, font, wrapped_lines
    
    # 如果最小字体仍然放不下，返回最小字体
    font = get_font(min_font_size)
    _, wrapped_lines = calculate_text_height(text, font, available_width, line_spacing=5)
    return min_font_size, font, wrapped_lines


def create_text_panel(text, panel_width, panel_height, bg_color=(30, 30, 30), text_color=(255, 255, 255)):
    """
    创建包含文字的面板，自动调整字体大小以完整显示文本
    """
    # 创建深色背景面板
    panel = Image.new('RGB', (panel_width, panel_height), bg_color)
    draw = ImageDraw.Draw(panel)
    
    margin = 10
    title_height = 35
    
    # 找到最优字体大小
    font_size, font, wrapped_lines = find_optimal_font_size(
        text, panel_width, panel_height, 
        margin=margin, title_height=title_height,
        min_font_size=10, max_font_size=22
    )
    
    # 获取标题字体（稍大一点）
    title_font = get_font(min(font_size + 2, 24))
    
    # 添加标题
    title = "Instruction:"
    draw.text((margin, margin), title, font=title_font, fill=(100, 200, 255))
    
    # 绘制换行后的文本
    bbox = font.getbbox("Ag")
    line_height = (bbox[3] - bbox[1]) + 5
    y_position = title_height
    
    for line in wrapped_lines:
        draw.text((margin, y_position), line, font=font, fill=text_color)
        y_position += line_height
    
    return np.array(panel)


def load_episode_data(dataset_path, key):
    """
    根据 key 加载 episode 数据
    key 格式: scan_episode_id (例如: scan_05_0)
    """
    splits = key.split('_')
    if len(splits) == 2:
        scan = splits[0]
        trajectory = splits[1]
    elif len(splits) == 3:
        # gru-vln10 格式: scan_05_0
        scan = splits[0] + '_' + splits[1]
        trajectory = splits[2]
    else:
        raise ValueError(f"Invalid key format: {key}")
    
    trajectory_path = os.path.join(dataset_path, scan, trajectory)
    
    if not os.path.exists(trajectory_path):
        raise FileNotFoundError(f"Trajectory path not found: {trajectory_path}")
    
    # 读取 RGB 数据
    rgb_path = os.path.join(trajectory_path, "videos/chunk-000/observation.images.rgb/rgb.npy")
    if not os.path.exists(rgb_path):
        raise FileNotFoundError(f"RGB data not found: {rgb_path}")
    
    rgb_data = np.load(rgb_path)
    
    # 读取 episodes.jsonl 获取 instruction
    json_path = os.path.join(trajectory_path, "meta/episodes.jsonl")
    instructions = []
    finish_status = None
    fail_reason = None
    
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            for line in f:
                try:
                    json_data = json.loads(line.strip())
                    if 'instruction_text' in json_data:
                        instructions.append(json_data['instruction_text'])
                    finish_status = json_data.get('finish_status', None)
                    fail_reason = json_data.get('fail_reason', None)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
    
    # 合并所有 instruction
    instruction_text = " | ".join(instructions) if instructions else "No instruction available"
    
    return {
        'rgb': rgb_data,
        'instruction': instruction_text,
        'finish_status': finish_status,
        'fail_reason': fail_reason,
        'trajectory_path': trajectory_path
    }


def visualize_episode(dataset_path, key, output_dir, fps=10, panel_width=400):
    """
    可视化 episode 并保存为视频
    """
    print(f"Loading episode: {key}")
    data = load_episode_data(dataset_path, key)
    
    rgb_frames = data['rgb']
    instruction = data['instruction']
    finish_status = data['finish_status']
    fail_reason = data['fail_reason']
    
    num_frames, height, width, channels = rgb_frames.shape
    print(f"Episode info:")
    print(f"  - Frames: {num_frames}")
    print(f"  - Frame size: {width}x{height}")
    print(f"  - Instruction: {instruction}")
    print(f"  - Finish status: {finish_status}")
    print(f"  - Fail reason: {fail_reason}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 输出视频路径
    output_video_path = os.path.join(output_dir, f"{key.replace('/', '_')}_vis.mp4")
    
    # 计算带文字面板的总宽度
    total_width = width + panel_width
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (total_width, height))
    
    # 构建完整的显示文本
    display_text = instruction
    # if finish_status is not None:
    #     display_text += f"\n\n[Status: {finish_status}]"
    # if fail_reason:
    #     display_text += f"\n[Fail reason: {fail_reason}]"
    
    print(f"Generating video with {num_frames} frames...")
    
    for i in range(num_frames):
        # 获取当前帧的 RGB
        frame_rgb = rgb_frames[i]
        
        # 创建文字面板，添加帧计数
        frame_text = f"Frame: {i+1}/{num_frames}\n\n{display_text}"
        text_panel = create_text_panel(frame_text, panel_width, height)
        
        # 将 RGB 转换为 BGR (OpenCV 格式)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        text_panel_bgr = cv2.cvtColor(text_panel, cv2.COLOR_RGB2BGR)
        
        # 水平拼接帧和文字面板
        combined_frame = np.hstack([frame_bgr, text_panel_bgr])
        
        # 写入视频
        video_writer.write(combined_frame)
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{num_frames} frames")
    
    video_writer.release()
    print(f"\nVideo saved to: {output_video_path}")
    
    return output_video_path


def list_available_keys(dataset_path):
    """
    列出数据集中所有可用的 key
    """
    keys = []
    for scan in os.listdir(dataset_path):
        scan_path = os.path.join(dataset_path, scan)
        if not os.path.isdir(scan_path):
            continue
        for trajectory in os.listdir(scan_path):
            trajectory_path = os.path.join(scan_path, trajectory)
            if not os.path.isdir(trajectory_path):
                continue
            keys.append(f"{scan}_{trajectory}")
    return sorted(keys)


def main():
    parser = argparse.ArgumentParser(description="Visualize LeRobot dataset episodes")
    parser.add_argument("--dataset_path", type=str, default="data/vln_pe/traj_data/gruvln10",
                        help="Path to the LeRobot dataset")
    parser.add_argument("--key", type=str, default=None,
                        help="Episode key in format 'scan_episode_id' (e.g., 'scan_05_0')")
    parser.add_argument("--output_dir", type=str, default="vis_output",
                        help="Output directory for videos")
    parser.add_argument("--fps", type=int, default=10,
                        help="Frames per second for the output video")
    parser.add_argument("--panel_width", type=int, default=400,
                        help="Width of the text panel")
    parser.add_argument("--list_keys", action="store_true",
                        help="List all available episode keys")
    
    args = parser.parse_args()
    
    if args.list_keys:
        print("Available episode keys:")
        keys = list_available_keys(args.dataset_path)
        for key in keys:
            print(f"  {key}")
        print(f"\nTotal: {len(keys)} episodes")
        return
    
    if args.key is None:
        print("Error: Please provide an episode key using --key argument")
        print("Use --list_keys to see all available episode keys")
        return
    
    # 可视化指定的 episode
    visualize_episode(
        dataset_path=args.dataset_path,
        key=args.key,
        output_dir=args.output_dir,
        fps=args.fps,
        panel_width=args.panel_width
    )


if __name__ == "__main__":
    main()

