"""
LMDB 数据可视化脚本
读取 sample_data.lmdb，根据 epId_trajId 可视化其中存储的 RGB 信息（逐张存储）
"""
import os
import argparse
import lmdb
import msgpack_numpy
import numpy as np
from PIL import Image


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
):
    """
    可视化指定 key 对应的 RGB 数据
    
    Args:
        lmdb_path: LMDB 数据库路径
        key: episode/trajectory 的 key (epId_trajId 或 trajId)
        output_dir: 输出目录
        camera_name: 相机名称，默认 pano_camera_0
    """
    # 创建输出目录
    save_dir = os.path.join(output_dir, key)
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
    print(f"Key: {key}")
    print(f"RGB 形状: {rgb_data.shape}")
    print(f"RGB 数据类型: {rgb_data.dtype}")
    print(f"Finish status: {value.get('finish_status', 'N/A')}")
    print(f"Fail reason: {value.get('fail_reason', 'N/A')}")
    print(f"保存目录: {save_dir}")
    
    # 逐张保存 RGB 图像
    num_frames = len(rgb_data)
    print(f"共 {num_frames} 帧图像")
    
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
        help="要可视化的 key (epId_trajId 或 trajId)",
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
        return
    
    visualize_rgb(
        lmdb_path=args.lmdb_path,
        key=args.key,
        output_dir=args.output_dir,
        camera_name=args.camera_name,
    )


if __name__ == "__main__":
    main()

