"""
LMDB 数据可视化脚本
读取 sample_data.lmdb，根据 epId_trajId 可视化其中存储的 RGB 信息
支持保存为图片序列或视频，并可显示 instruction
"""
import os
import argparse
import lmdb
import msgpack_numpy
import numpy as np
import gzip
import json
from PIL import Image, ImageDraw, ImageFont
import cv2


def get_font(font_size):
    """获取字体，优先尝试支持中文的字体"""
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
    """将文本换行以适应指定宽度"""
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
    """计算文本在给定宽度下换行后的总高度"""
    wrapped_lines = wrap_text(text, font, max_width)
    bbox = font.getbbox("Ag")
    line_height = (bbox[3] - bbox[1]) + line_spacing
    return len(wrapped_lines) * line_height, wrapped_lines


def find_optimal_font_size(text, panel_width, panel_height, margin=10, title_height=35, min_font_size=10, max_font_size=24):
    """找到能让文本完全显示的最大字体大小"""
    available_width = panel_width - 2 * margin
    available_height = panel_height - title_height - 2 * margin
    
    for font_size in range(max_font_size, min_font_size - 1, -1):
        font = get_font(font_size)
        text_height, wrapped_lines = calculate_text_height(text, font, available_width, line_spacing=5)
        
        if text_height <= available_height:
            return font_size, font, wrapped_lines
    
    font = get_font(min_font_size)
    _, wrapped_lines = calculate_text_height(text, font, available_width, line_spacing=5)
    return min_font_size, font, wrapped_lines


def create_text_panel(text, panel_width, panel_height, bg_color=(30, 30, 30), text_color=(255, 255, 255)):
    """创建包含文字的面板，自动调整字体大小以完整显示文本"""
    panel = Image.new('RGB', (panel_width, panel_height), bg_color)
    draw = ImageDraw.Draw(panel)
    
    margin = 10
    title_height = 35
    
    font_size, font, wrapped_lines = find_optimal_font_size(
        text, panel_width, panel_height, 
        margin=margin, title_height=title_height,
        min_font_size=10, max_font_size=22
    )
    
    title_font = get_font(min(font_size + 2, 24))
    
    title = "Instruction:"
    draw.text((margin, margin), title, font=title_font, fill=(100, 200, 255))
    
    bbox = font.getbbox("Ag")
    line_height = (bbox[3] - bbox[1]) + 5
    y_position = title_height
    
    for line in wrapped_lines:
        draw.text((margin, y_position), line, font=font, fill=text_color)
        y_position += line_height
    
    return np.array(panel)


def load_data_from_json(data_json_file):
    """
    从 json.gz 文件加载数据，返回：
    - episodeId_to_trajId: episode_id -> trajectory_id 的映射
    - traj_to_instructions: trajectory_id -> instructions 列表的映射
    """
    if not os.path.exists(data_json_file):
        print(f"警告: 数据文件不存在: {data_json_file}")
        return {}, {}
    
    try:
        if data_json_file.endswith('.gz'):
            with gzip.open(data_json_file, 'rt', encoding='utf-8') as f:
                load_data = json.load(f)['episodes']
        else:
            with open(data_json_file, 'r', encoding='utf-8') as f:
                load_data = json.load(f)['episodes']
    except Exception as e:
        print(f"警告: 无法加载数据文件: {e}")
        return {}, {}
    
    # 建立 episode_id -> trajectory_id 的映射
    episodeId_to_trajId = {}
    # 建立 trajectory_id -> instructions 的映射
    traj_to_instructions = {}
    
    for episode in load_data:
        episode_id = str(episode['episode_id'])
        traj_id = str(episode['trajectory_id'])
        
        episodeId_to_trajId[episode_id] = traj_id
        
        if traj_id not in traj_to_instructions:
            traj_to_instructions[traj_id] = []
        if 'instruction' in episode and 'instruction_text' in episode['instruction']:
            traj_to_instructions[traj_id].append(episode['instruction']['instruction_text'])
    
    return episodeId_to_trajId, traj_to_instructions


def parse_key(key):
    """
    解析 key，支持两种格式：
    1. scan_05_0 格式 (scan_episodeId) -> 返回 (scan, episode_id)
    2. 纯数字格式 (trajectory_id) -> 返回 (None, key)
    """
    splits = key.split('_')
    if len(splits) == 3:
        # scan_05_0 格式
        scan = splits[0] + '_' + splits[1]
        episode_id = splits[2]
        return scan, episode_id
    elif len(splits) == 2 and splits[0].startswith('scan'):
        # scan_0 格式（如果 scan 不带数字）
        scan = splits[0]
        episode_id = splits[1]
        return scan, episode_id
    else:
        # 纯数字或其他格式，当作 trajectory_id
        return None, key


def list_all_keys(lmdb_path: str):
    """列出 LMDB 中所有的 key"""
    lmdb_env = lmdb.open(lmdb_path, readonly=True, lock=False, max_dbs=1)
    with lmdb_env.begin() as txn:
        cursor = txn.cursor()
        all_keys = [key.decode() for key, _ in cursor]
    lmdb_env.close()
    return all_keys


def visualize_rgb(
    lmdb_path: str,
    key: str,
    output_dir: str,
    camera_name: str = "pano_camera_0",
    save_video: bool = False,
    fps: int = 10,
    instruction: str = None,
    panel_width: int = 400,
    display_key: str = None,
):
    """
    可视化指定 key 对应的 RGB 数据
    
    Args:
        lmdb_path: LMDB 数据库路径
        key: LMDB 中的 trajectory_id
        output_dir: 输出目录
        camera_name: 相机名称，默认 pano_camera_0
        save_video: 是否保存为视频
        fps: 视频帧率
        instruction: instruction 文本
        panel_width: 文字面板宽度
        display_key: 用于显示和文件名的 key（如 scan_05_0）
    """
    # 使用 display_key 作为目录名和文件名
    if display_key is None:
        display_key = key
    
    # 创建输出目录
    save_dir = os.path.join(output_dir, display_key)
    os.makedirs(save_dir, exist_ok=True)
    
    # 打开 LMDB 数据库
    lmdb_env = lmdb.open(lmdb_path, readonly=True, lock=False, max_dbs=1)
    
    with lmdb_env.begin() as txn:
        value_packed = txn.get(key.encode())
        if value_packed is None:
            print(f"错误: key '{key}' 在 LMDB 中未找到")
            lmdb_env.close()
            return False
        
        value = msgpack_numpy.unpackb(value_packed)
    
    lmdb_env.close()
    
    # 提取 RGB 数据
    try:
        episode_data = value['episode_data']
        camera_info = episode_data['camera_info']
        
        if camera_name not in camera_info:
            print(f"错误: 相机 '{camera_name}' 不存在")
            print(f"可用的相机: {list(camera_info.keys())}")
            return False
        
        rgb_data = camera_info[camera_name]['rgb']
    except KeyError as e:
        print(f"错误: 无法解析数据结构, 缺少 key: {e}")
        return False
    
    # 打印基本信息
    finish_status = value.get('finish_status', 'N/A')
    fail_reason = value.get('fail_reason', 'N/A')
    
    print(f"Key: {display_key} (LMDB key: {key})")
    print(f"RGB 形状: {rgb_data.shape}")
    print(f"RGB 数据类型: {rgb_data.dtype}")
    print(f"Finish status: {finish_status}")
    print(f"Fail reason: {fail_reason}")
    if instruction:
        print(f"Instruction: {instruction}")
    print(f"保存目录: {save_dir}")
    
    num_frames = len(rgb_data)
    height, width = rgb_data.shape[1], rgb_data.shape[2]
    print(f"共 {num_frames} 帧图像, 尺寸: {width}x{height}")
    
    # 准备视频写入器
    video_writer = None
    if save_video:
        if instruction:
            total_width = width + panel_width
        else:
            total_width = width
        video_path = os.path.join(output_dir, f"lmdb_{display_key}_vis.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (total_width, height))
        print(f"将保存视频到: {video_path}")
    
    # 逐帧处理
    for idx, rgb_frame in enumerate(rgb_data):
        # 确保数据是 uint8 格式
        if rgb_frame.dtype != np.uint8:
            if rgb_frame.max() <= 1.0:
                rgb_frame = (rgb_frame * 255).astype(np.uint8)
            else:
                rgb_frame = rgb_frame.astype(np.uint8)
        
        # 保存为 PNG 图像
        img = Image.fromarray(rgb_frame)
        img_path = os.path.join(save_dir, f"frame_{idx:06d}.png")
        img.save(img_path)
        
        # 如果需要保存视频
        if save_video:
            frame_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            
            if instruction:
                # 创建带帧信息的文字面板
                frame_text = f"Frame: {idx+1}/{num_frames}\n\n{instruction}"
                text_panel = create_text_panel(frame_text, panel_width, height)
                text_panel_bgr = cv2.cvtColor(text_panel, cv2.COLOR_RGB2BGR)
                combined_frame = np.hstack([frame_bgr, text_panel_bgr])
            else:
                combined_frame = frame_bgr
            
            video_writer.write(combined_frame)
        
        if (idx + 1) % 50 == 0:
            print(f"  已处理 {idx + 1}/{num_frames} 帧")
    
    if video_writer:
        video_writer.release()
        print(f"视频已保存到: {video_path}")
    
    print(f"成功保存 {num_frames} 张图像到 {save_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description="可视化 LMDB 中的 RGB 数据")
    parser.add_argument(
        "--lmdb_path",
        type=str,
        default="/cpfs/shared/simulation/vln-pe/20250314_sample_grutopia_vln10_controller/sample_data.lmdb",
        help="LMDB 数据库路径",
    )
    parser.add_argument(
        "--key",
        type=str,
        required=False,
        help="要可视化的 key，支持 scan_05_0 格式 (scan_episodeId) 或纯 trajectory_id",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./vis_output",
        help="输出目录",
    )
    parser.add_argument(
        "--camera_name",
        type=str,
        default="pano_camera_0",
        help="相机名称",
    )
    parser.add_argument(
        "--list_keys",
        action="store_true",
        help="列出所有可用的 key",
    )
    parser.add_argument(
        "--list_limit",
        type=int,
        default=20,
        help="列出 key 时的最大数量限制",
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="是否保存为视频",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="视频帧率",
    )
    parser.add_argument(
        "--data_json",
        type=str,
        default="data/data_old/grutopia10/train/train.json.gz",
        help="包含 instruction 信息的 json.gz 文件路径",
    )
    parser.add_argument(
        "--panel_width",
        type=int,
        default=400,
        help="instruction 面板宽度",
    )
    parser.add_argument(
        "--no_instruction",
        action="store_true",
        help="不显示 instruction",
    )
    
    args = parser.parse_args()
    
    # 检查 LMDB 路径是否存在
    if not os.path.exists(args.lmdb_path):
        print(f"错误: LMDB 路径不存在: {args.lmdb_path}")
        return
    
    # 如果只是列出 keys
    if args.list_keys:
        all_keys = list_all_keys(args.lmdb_path)
        print(f"LMDB 中共有 {len(all_keys)} 个 key")
        print(f"前 {min(args.list_limit, len(all_keys))} 个 key:")
        for i, key in enumerate(all_keys[:args.list_limit]):
            print(f"  [{i+1}] {key}")
        return
    
    # 可视化指定的 key
    if args.key is None:
        print("请指定 --key 参数，或使用 --list_keys 查看所有可用的 key")
        print("支持格式: scan_05_0 (scan_episodeId) 或纯 trajectory_id")
        return
    
    # 加载数据映射
    episodeId_to_trajId, traj_to_instructions = load_data_from_json(args.data_json)
    
    # 解析 key，转换为 trajectory_id
    scan, episode_or_traj_id = parse_key(args.key)
    
    if scan is not None:
        # scan_05_0 格式，需要查找对应的 trajectory_id
        if episode_or_traj_id in episodeId_to_trajId:
            lmdb_key = episodeId_to_trajId[episode_or_traj_id]
            print(f"解析 key: {args.key} -> episode_id: {episode_or_traj_id} -> trajectory_id: {lmdb_key}")
        else:
            print(f"错误: episode_id '{episode_or_traj_id}' 在数据文件中未找到")
            print("请确保 --data_json 参数指向正确的数据文件")
            return
    else:
        # 纯 trajectory_id 格式
        lmdb_key = episode_or_traj_id
        print(f"使用 trajectory_id: {lmdb_key}")
    
    # 加载 instruction
    instruction = None
    if not args.no_instruction:
        if lmdb_key in traj_to_instructions:
            instructions_list = traj_to_instructions[lmdb_key]
            instruction = " | ".join(instructions_list) if instructions_list else None
        if instruction:
            print(f"找到 instruction: {instruction[:100]}...")
        else:
            print("未找到对应的 instruction")
    
    visualize_rgb(
        lmdb_path=args.lmdb_path,
        key=lmdb_key,
        output_dir=args.output_dir,
        camera_name=args.camera_name,
        save_video=args.save_video,
        fps=args.fps,
        instruction=instruction,
        panel_width=args.panel_width,
        display_key=args.key,  # 用于显示的原始 key
    )


if __name__ == "__main__":
    main()
