#!/usr/bin/env python
"""
LMDB导航数据格式转换脚本 - 完全符合LeRobot v3.0标准
"""
import json
import shutil
import os
import tqdm
import numpy as np
import datasets
import threading
import sys
import datetime
from pathlib import Path
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple
from datasets import concatenate_datasets
from lerobot.datasets.utils import (
    embed_images, 
    hf_transform_to_torch,
    flatten_dict,
    write_info,
    write_stats,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats, sample_indices
import lmdb
import msgpack_numpy
import torch
from collections import defaultdict
import gzip

# --- 全局变量定义 ---
LEROBOT_HOME = Path(os.environ.get("LEROBOT_HOME", "/shared/smartbot/vln-pe/gr10_data/grutopia10/lerobot_data"))

# --- 预定义基础 Features ---
def create_base_features():
    """创建基础特征字典"""
    return {
        # Camera Info
        "observation.camera_position": {
            "dtype": "float64",
            "shape": (3,),
            "names": ["x", "y", "z"]
        },
        "observation.camera_orientation": {
            "dtype": "float64",
            "shape": (4,),
            "names": ["qx", "qy", "qz", "qw"]
        },
        "observation.camera_yaw": {
            "dtype": "float64",
            "shape": (1,),
            "names": ["x"]
        },
        # Robot Info
        "observation.robot_position": {
            "dtype": "float64",
            "shape": (3,),
            "names": ["x", "y", "z"]
        },
        "observation.robot_orientation": {
            "dtype": "float64",
            "shape": (4,),
            "names": ["qx", "qy", "qz", "qw"]
        },
        "observation.robot_yaw": {
            "dtype": "float64",
            "shape": (1,),
            "names": ["x"]
        },
        "observation.progress": {
            "dtype": "float64",
            "shape": (1,),
            "names": ["progress"]
        },
        "observation.step": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["step"]
        },
        # Action
        "observation.action": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["action"]
        },
    }

def ensure_numpy_array(data, expected_dtype=None, expand_dims=False):
    """确保数据是numpy数组"""
    if data is None:
        return None
        
    if not isinstance(data, np.ndarray):
        try:
            if isinstance(data, list):
                data = np.array(data, dtype=expected_dtype)
            else:
                data = np.array(data, dtype=expected_dtype)
        except Exception as e:
            logger.warning(f"无法将数据转换为numpy数组: {e}")
            return None
    
    if expand_dims and data.ndim == 0:
        data = np.expand_dims(data, axis=0)
        
    if expected_dtype is not None and data.dtype != expected_dtype:
        try:
            data = data.astype(expected_dtype)
        except Exception as e:
            logger.warning(f"无法将数据类型转换为 {expected_dtype}: {e}")
            
    return data

def compute_episode_stats(episode_buffer: dict, features: dict) -> dict:
    """计算episode的统计信息"""
    ep_stats = {}
    for key, data_list in episode_buffer.items():
        if key in ["index", "episode_index", "frame_index", "timestamp", "task", "task_index", "size"]:
            continue
        if key not in features:
            continue

        feature_info = features[key]
        if feature_info["dtype"] == "string":
            continue

        try:
            if isinstance(data_list, list):
                if feature_info["dtype"] == "image" and data_list and isinstance(data_list[0], str):
                     continue
                data_array = np.stack(data_list)
            else:
                data_array = np.asarray(data_list)
        except (ValueError, np.exceptions.AxisError) as e:
            logger.warning(f"无法将特征 {key} 转换为numpy数组: {e}")
            continue

        try:
            if feature_info["dtype"] in ["image"]:
                sampled_indices = sample_indices(len(data_array))
                if len(sampled_indices) == 0:
                    continue
                ep_ft_array = data_array[sampled_indices]
                value_norm = 1.0 if "depth" in key else 255.0
                if value_norm != 1.0:
                    ep_ft_array = ep_ft_array.astype(np.float64) / value_norm
                axes_to_reduce = tuple(range(data_array.ndim - 1))
                keepdims = True
            else:
                ep_ft_array = data_array
                axes_to_reduce = 0
                keepdims = data_array.ndim <= 2

            ep_stats[key] = get_feature_stats(ep_ft_array, axis=axes_to_reduce, keepdims=keepdims)

            if feature_info["dtype"] in ["image"] and "count" in ep_stats[key]:
                ep_stats[key].pop("count", None)
                for stat_name, stat_val in ep_stats[key].items():
                    if isinstance(stat_val, np.ndarray):
                        ep_stats[key][stat_name] = np.squeeze(stat_val, axis=axes_to_reduce if keepdims else None)

        except Exception as e:
            logger.warning(f"计算特征 {key} 的统计信息失败: {e}")
            continue
    return ep_stats

# --- 自定义 LeRobotDataset 子类 ---
class NavDataset(LeRobotDataset):
    """符合LeRobot v3.0标准的导航数据集类"""

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        features: dict,
        root: str | Path | None = None,
        robot_type: str | None = None,
        use_videos: bool = False,
        tolerance_s: float = 1e-4,
        image_writer_processes: int = 0,
        image_writer_threads: int = 4,
        video_backend: str | None = None,
    ) -> "NavDataset":
        """创建符合v3.0标准的导航数据集"""
        from lerobot.datasets.video_utils import get_safe_default_codec
        
        # 使用基类的create方法创建metadata
        obj = cls.__new__(cls)
        obj.meta = LeRobotDatasetMetadata.create(
            repo_id=repo_id,
            fps=fps,
            robot_type=robot_type,
            features=features,
            root=root,
            use_videos=use_videos,
        )
        
        # 自定义 data_path 格式：使用 episode_index 而不是 file_index
        obj.meta.info["data_path"] = "data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet"
        
        obj.repo_id = obj.meta.repo_id
        obj.root = obj.meta.root
        obj.revision = None
        obj.tolerance_s = tolerance_s
        obj.image_writer = None

        if image_writer_processes or image_writer_threads:
            obj.start_image_writer(image_writer_processes, image_writer_threads)

        obj.episode_buffer = obj.create_episode_buffer()
        obj.episodes = None
        obj.hf_dataset = obj.create_hf_dataset()
        obj.image_transforms = None
        obj.delta_timestamps = None
        obj.delta_indices = None
        obj.episode_data_index = None
        obj.video_backend = video_backend if video_backend is not None else get_safe_default_codec()
        return obj

    def add_frame(self, frame: dict, task: str, timestamp: float | None = None) -> None:
        """添加帧数据到episode_buffer"""
        # 确保所有torch张量转换为numpy数组
        for name in frame:
            if isinstance(frame[name], torch.Tensor):
                frame[name] = frame[name].numpy()
            elif isinstance(frame[name], list):
                frame[name] = np.array(frame[name])

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        # 添加frame_index和timestamp
        frame_index = self.episode_buffer["size"]
        if timestamp is None:
            timestamp = frame_index / self.fps
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)
        self.episode_buffer["task"].append(task)

        # 添加帧特征到episode_buffer
        for key, value in frame.items():
            self.episode_buffer[key].append(value)

        self.episode_buffer["size"] += 1

    def save_episode(self, episode_index: int, episode_metadata: dict = None) -> None:
        """保存episode数据 - 符合v3.0标准"""
        if not self.episode_buffer or self.episode_buffer["size"] == 0:
            logger.warning("Episode buffer为空，跳过保存")
            return

        episode_buffer = self.episode_buffer
        episode_length = episode_buffer["size"]
        tasks = episode_buffer["task"]
        
        # 处理任务
        new_tasks = [task for task in tasks if not isinstance(task, dict)]
        if not new_tasks:
            logger.error("没有有效的任务数据")
            raise ValueError("没有有效的任务数据")
        episode_tasks = list(set(new_tasks))
        logger.debug(f"Episode {episode_index} 任务: {episode_tasks}")

        # 准备索引数据
        episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index, dtype=np.int64)

        # 处理任务索引
        # 使用save_episode_tasks来处理任务注册
        self.meta.save_episode_tasks(episode_tasks)
        episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in new_tasks])

        # 确保所有非图像特征都是正确形状的numpy数组
        for key, ft in self.features.items():
            if key in ["index", "episode_index", "task_index", "frame_index", "timestamp"] or ft["dtype"] in ["image", "video"]:
                continue
                
            if isinstance(episode_buffer[key], list):
                for i in range(len(episode_buffer[key])):
                    if not isinstance(episode_buffer[key][i], np.ndarray):
                        episode_buffer[key][i] = np.array(episode_buffer[key][i], dtype=ft["dtype"])
                
                try:
                    episode_buffer[key] = np.stack(episode_buffer[key])
                except ValueError as e:
                    logger.warning(f"堆叠特征 {key} 时出错: {e}")
                    max_shape = max([x.shape for x in episode_buffer[key]], key=lambda s: np.prod(s))
                    padded = []
                    for item in episode_buffer[key]:
                        pad_width = [(0, max_shape[i] - item.shape[i]) for i in range(len(item.shape))]
                        padded_item = np.pad(item, pad_width, mode='constant')
                        padded.append(padded_item)
                    episode_buffer[key] = np.stack(padded)
            
            # 确保形状正确
            if len(episode_buffer[key].shape) > 2:
                episode_buffer[key] = episode_buffer[key].reshape(episode_buffer[key].shape[0], -1)

        # 计算统计信息
        ep_stats = compute_episode_stats(episode_buffer, self.features)
        
        # 保存数据到parquet文件
        self._save_episode_table(episode_buffer, episode_index)
        
        # 准备episode元数据字典
        episode_dict = {
            "episode_index": episode_index,
            "tasks": episode_tasks,
            "length": episode_length,
        }
        
        # 添加自定义元数据
        if episode_metadata:
            episode_dict.update(episode_metadata)
        
        # 添加统计信息（扁平化）
        episode_dict.update(flatten_dict({"stats": ep_stats}))
        
        # 使用基类的方法保存元数据
        self.meta._save_episode_metadata(episode_dict)

        # 更新info
        self.meta.info["total_episodes"] += 1
        self.meta.info["total_frames"] += episode_length
        
        # 计算并更新total_chunks
        chunk = episode_index // self.meta.chunks_size
        if "total_chunks" not in self.meta.info:
            self.meta.info["total_chunks"] = 0
        if chunk >= self.meta.info["total_chunks"]:
            self.meta.info["total_chunks"] = chunk + 1

        self.meta.info["splits"] = {"train": f"0:{self.meta.info['total_episodes']}"}
        
        write_info(self.meta.info, self.meta.root)

        # 更新统计信息
        self.meta.stats = aggregate_stats([self.meta.stats, ep_stats]) if self.meta.stats else ep_stats
        write_stats(self.meta.stats, self.meta.root)

        # 保存自定义元数据到JSONL文件
        if episode_metadata and 'instructions' in episode_metadata:
            episodes_file = self.root / "meta" / "episodes.jsonl"
            episodes_file.parent.mkdir(parents=True, exist_ok=True)
            with open(episodes_file, 'a', encoding='utf-8') as f:
                for instruction_dict in episode_metadata['instructions']:
                    f.write(json.dumps(instruction_dict, ensure_ascii=False) + "\n")

        # 重置buffer
        self.episode_buffer = self.create_episode_buffer()

    def _save_episode_table(self, episode_buffer: dict, episode_index: int) -> None:
        """保存episode数据到parquet文件"""
        episode_dict = {key: episode_buffer[key] for key in self.hf_features if key in episode_buffer}
        ep_dataset = datasets.Dataset.from_dict(episode_dict, features=self.hf_features, split="train")
        ep_dataset = embed_images(ep_dataset)
        self.hf_dataset = concatenate_datasets([self.hf_dataset, ep_dataset])
        self.hf_dataset.set_transform(hf_transform_to_torch)
        
        # 计算chunk索引，使用episode_index作为文件名
        chunk_idx = episode_index // self.meta.chunks_size
        ep_data_path = self.root / self.meta.data_path.format(chunk_index=chunk_idx, episode_index=0)
        ep_data_path.parent.mkdir(parents=True, exist_ok=True)
        ep_dataset.to_parquet(ep_data_path)
        
    def finalize(self):
        """完成数据集创建，刷新所有缓冲"""
        # 等待图像写入完成
        if hasattr(self, '_wait_image_writer'):
            self._wait_image_writer()
        
        # 刷新元数据缓冲
        if hasattr(self.meta, '_flush_metadata_buffer'):
            self.meta._flush_metadata_buffer()
        
        logger.info(f"数据集创建完成: {self.root}")
        logger.info(f"总episodes: {self.meta.info['total_episodes']}")
        logger.info(f"总frames: {self.meta.info['total_frames']}")

def load_gather_data_gruvln10_new(data_json: Path) -> Tuple[Dict, Dict, Dict]:
    """加载train.json.gz并建立映射"""
    try:
        with gzip.open(data_json, 'rt', encoding='utf-8') as f:
            data = json.load(f)['episodes']
        episode_to_trajectory = {}
        scene_to_trajectories = {}
        trajectory_to_episodes = defaultdict(list)
        for episode in data:
            if episode['scan'] not in scene_to_trajectories:
                scene_to_trajectories[episode['scan']] = []
            episode_id = str(episode['episode_id'])
            trajectory_id = str(episode['trajectory_id'])
            episode_to_trajectory[episode_id] = episode
            scene_to_trajectories[episode['scan']].append(episode)
            trajectory_to_episodes[trajectory_id].append(episode)
        logger.info(f"成功加载 {data_json}，包含 {len(episode_to_trajectory)} 个 episodes.")
        return episode_to_trajectory, scene_to_trajectories, trajectory_to_episodes
    except Exception as e:
        logger.error(f"加载 json.gz 文件失败 {data_json}: {e}")
        raise

def process_trajectory_group(
    lmdb_env,
    group_key: Tuple[str, str, str],
    lmdb_keys_in_group: List[str],
    episode_to_trajectory: Dict,
    scene_to_trajectories: Dict,
    trajectory_to_episodes: Dict,
    repo_name: str,
    push_to_hub: bool,
):
    """处理一个完整的 trajectory group - 为每个episode创建单独的数据集"""
    thread_name = threading.current_thread().name
    scene_dataset, scene, trajectory_id = group_key
    logger.info(f"线程 [{thread_name}] 开始处理 trajectory: {scene_dataset}/{scene}/{trajectory_id}")

    # 获取该trajectory对应的所有episodes
    all_episodes_for_traj = trajectory_to_episodes.get(str(trajectory_id), [])
    if not all_episodes_for_traj:
        return (group_key, False, f"Trajectory {trajectory_id} 没有对应的episodes")
    
    logger.info(f"Trajectory {trajectory_id} 包含 {len(all_episodes_for_traj)} 个 episodes")
    
    # 选择代表性LMDB key（使用trajectory_id作为key）
    representative_key = lmdb_keys_in_group[0]

    # 读取trajectory数据（一次）
    with lmdb_env.begin() as txn:
        value_packed = txn.get(representative_key.encode())
        if value_packed is None:
            return (group_key, False, f"Key {representative_key} 未找到")
        value = msgpack_numpy.unpackb(value_packed)

    if 'episode_data' not in value:
        return (group_key, False, "缺少 episode_data")

    trajectory_data = value['episode_data']  # 这是trajectory的数据
    finish_status = value.get('finish_status', 'unknown')
    fail_reason = value.get('fail_reason', 'unknown')
    
    # 排序episodes
    all_episodes_for_traj.sort(key=lambda x: x['episode_id'])

    # 创建key映射 (注意：flatten_dict使用'/'作为分隔符)
    key_mapping = {
        'camera_info/pano_camera_0/rgb': 'observation.rgb',
        'camera_info/pano_camera_0/depth': 'observation.depth',
        'camera_info/pano_camera_0/position': 'observation.camera_position',
        'camera_info/pano_camera_0/orientation': 'observation.camera_orientation',
        'camera_info/pano_camera_0/yaw': 'observation.camera_yaw',
        'robot_info/position': 'observation.robot_position',
        'robot_info/orientation': 'observation.robot_orientation',
        'robot_info/yaw': 'observation.robot_yaw',
        'progress': 'observation.progress',
        'step': 'observation.step',
        'action': 'observation.action',
    }

    # 扁平化trajectory数据（一次性处理，所有episode共享）
    flattened_data = flatten_dict(trajectory_data)
    
    # 调试：打印前几个键以验证分隔符
    sample_keys = list(flattened_data.keys())[:5]
    logger.debug(f"扁平化后的样例键: {sample_keys}")
    
    remapped_data = {}
    unmapped_keys = []
    for k, v in flattened_data.items():
        if k in key_mapping:
            remapped_data[key_mapping[k]] = v
        else:
            remapped_data[k] = v
            if k not in ['camera_info/pano_camera_0/rgb', 'camera_info/pano_camera_0/depth']:
                unmapped_keys.append(k)
    
    if unmapped_keys:
        logger.debug(f"未映射的键 (前10个): {unmapped_keys[:10]}")

    # 创建特征字典
    features = create_base_features()

    # 处理数据并准备帧
    lerobot_frames = []
    frame_count = 0

    # 确定帧数
    for key in ['observation.camera_yaw', 'observation.robot_yaw', 'observation.progress', 'observation.step']:
        if key in remapped_data:
            data = ensure_numpy_array(remapped_data[key])
            if data is not None and data.ndim > 0:
                frame_count = max(frame_count, data.shape[0])

    if frame_count == 0:
        return (group_key, False, "无法确定帧数")

    # 处理每个特征
    processed_data = {}
    for key, raw_data in remapped_data.items():
        if key in ['observation.rgb', 'observation.depth']:
            continue
            
        # 处理action数据
        if key == "observation.action":
            if isinstance(raw_data, list):
                new_data = []
                for action in raw_data:
                    if isinstance(action, list):
                        new_data.append(action[0])
                    else:
                        new_data.append(action)
                raw_data = new_data
        
        # 确定预期数据类型
        expected_dtype = None
        if key in features:
            dtype_str = features[key]['dtype']
            if dtype_str == 'float32':
                expected_dtype = np.float32
            elif dtype_str == 'float64':
                expected_dtype = np.float64
            elif dtype_str == 'int64':
                expected_dtype = np.int64
        
        # 转换为numpy数组
        data_array = ensure_numpy_array(raw_data, expected_dtype, expand_dims=True)
        
        if data_array is not None:
            # 处理维度
            if data_array.ndim == 1:
                if key in features and 'shape' in features[key]:
                    target_shape = features[key]['shape']
                    if len(target_shape) == 1 and target_shape[0] == 1:
                        data_array = np.expand_dims(data_array, axis=-1)
            
            # 调整长度以匹配帧数
            if data_array.shape[0] != frame_count:
                if data_array.shape[0] < frame_count:
                    repeat_factor = (frame_count // data_array.shape[0]) + 1
                    data_array = np.repeat(data_array, repeat_factor, axis=0)[:frame_count]
                else:
                    data_array = data_array[:frame_count]
            
            processed_data[key] = data_array

    # 为每个episode创建单独的数据集
    success_count = 0
    failed_episodes = []
    
    for ep_info in all_episodes_for_traj:
        episode_id = str(ep_info['episode_id'])
        instruction_text = ep_info['instruction']['instruction_text']
        instruction_tokens = ep_info['instruction'].get('instruction_tokens', [])
        
        logger.info(f"处理 episode {episode_id}: {instruction_text[:50]}...")
        
        # 为每个episode创建独立的输出目录
        output_path = LEROBOT_HOME / repo_name / scene_dataset / scene / episode_id
        
        # 清理并创建目录
        if output_path.exists():
            try:
                shutil.rmtree(output_path)
            except Exception as e:
                logger.warning(f"清理目录 {output_path} 失败: {e}")
        
        try:
            # 创建数据集
            dataset = NavDataset.create(
                repo_id=repo_name,
                root=output_path,
                robot_type="navigation_robot",
                fps=6,
                use_videos=False,
                features=features,
            )
            
            # 添加帧数据（使用该episode的instruction作为task）
            task_text = instruction_text
            
            for i in range(frame_count):
                frame_data = {}
                for key, data_array in processed_data.items():
                    if i < data_array.shape[0]:
                        frame_data[key] = data_array[i]
                    else:
                        # 使用默认值
                        default_shape = features.get(key, {}).get('shape', ())
                        default_dtype = features.get(key, {}).get('dtype', 'float64')
                        frame_data[key] = np.zeros(default_shape, dtype=default_dtype)
                
                dataset.add_frame(frame=frame_data, task=task_text, timestamp=i/6.0)
            
            # 准备该episode的元数据
            episode_instructions = [{
                "episode_id": episode_id,
                "instruction_text": instruction_text,
                "instruction_tokens": instruction_tokens,
                "finish_status": finish_status,
                "fail_reason": fail_reason,
            }]
            
            episode_metadata = {
                'instructions': episode_instructions,
                'trajectory_id': trajectory_id,
                'episode_id': episode_id,
                'scene': scene,
                'scene_dataset': scene_dataset,
            }
            
            # 保存episode
            dataset.save_episode(episode_index=0, episode_metadata=episode_metadata)
            dataset.finalize()
            
            # 保存RGB和Depth为numpy文件
            try:
                if 'camera_info' in trajectory_data and 'pano_camera_0' in trajectory_data['camera_info']:
                    rgb = trajectory_data['camera_info']['pano_camera_0'].get('rgb')
                    if rgb is not None:
                        rgb_path = output_path / "videos" / "chunk-000" / "observation.images.rgb"
                        rgb_path.mkdir(parents=True, exist_ok=True)
                        np.save(rgb_path / 'rgb.npy', rgb)
                    
                    depth = trajectory_data['camera_info']['pano_camera_0'].get('depth')
                    if depth is not None:
                        depth_path = output_path / "videos" / "chunk-000" / "observation.images.depth"
                        depth_path.mkdir(parents=True, exist_ok=True)
                        np.save(depth_path / 'depth.npy', depth)
            except Exception as e:
                logger.warning(f"保存RGB/Depth numpy文件失败: {e}")
            
            success_count += 1
            logger.info(f"成功保存 episode {episode_id}")
            
        except Exception as e:
            logger.error(f"处理 episode {episode_id} 失败: {e}")
            failed_episodes.append(episode_id)
            continue

    # 返回处理结果
    if success_count > 0:
        message = f"成功处理 trajectory {trajectory_id}: {success_count}/{len(all_episodes_for_traj)} episodes, {frame_count} 帧"
        if failed_episodes:
            message += f", 失败的episodes: {failed_episodes}"
        logger.info(f"线程 [{thread_name}] {message}")
        return (group_key, True, message)
    else:
        message = f"Trajectory {trajectory_id} 所有episodes都处理失败"
        return (group_key, False, message)


def main(
    lmdb_path: str,
    gather_data_json: str,
    repo_name: str = "lerobot_v3",
    *,
    push_to_hub: bool = False,
    num_threads: int = 1,
    start_index: int = 0,
    end_index: int | None = None,
):
    """LMDB到LeRobot v3.0转换主函数"""

    log_dir = Path("/cpfs/user/wangliuyi/code/internnav_wangyukai/logs/vln_pe")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    end_str = "end" if end_index is None else str(end_index)
    log_file_path = log_dir / f"v3_s{start_index}-e{end_str}_{timestamp}.log"
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    logger.add(log_file_path, level="DEBUG", rotation="100 MB", encoding="utf-8")
    logger.info("="*50)
    logger.info("LeRobot v3.0 数据集转换脚本")
    logger.info(f"处理索引范围: [{start_index}, {end_index if end_index is not None else 'end'})")
    logger.info(f"日志文件: {log_file_path}")
    logger.info("="*50)

    lmdb_path_obj = Path(lmdb_path)
    gather_data_json_obj = Path(gather_data_json)

    # 加载LMDB keys
    logger.info("正在加载LMDB keys...")
    all_keys = []
    with lmdb.open(lmdb_path, readonly=True, lock=False) as lmdb_env:
        with lmdb_env.begin() as txn:
            cursor = txn.cursor()
            while cursor.next():
                all_keys.append(cursor.key().decode())
    
    if not all_keys:
        logger.warning("没有找到任何key。退出。")
        return
    logger.info(f"加载了 {len(all_keys)} 个key。")

    # 加载gather_data
    logger.info("正在加载 json.gz...")
    episode_to_trajectory, scene_to_trajectories, trajectory_to_episodes = load_gather_data_gruvln10_new(gather_data_json_obj)
    logger.info("json.gz 加载完成。")

    # 选择keys
    selected_keys = all_keys[start_index:end_index]
    if not selected_keys:
        logger.warning("在指定的索引范围内没有找到需要处理的key。退出。")
        return
    logger.info(f"选定 {len(selected_keys)} 个key进行处理。")

    # 分组
    logger.info("正在对keys进行分组...")
    trajectory_groups = {}

    for lmdb_key in selected_keys:
        episode_id = lmdb_key.split('_')[0] if '_' in lmdb_key else lmdb_key
        
        if episode_id not in trajectory_to_episodes:
            logger.warning(f"Key {lmdb_key} 在 gather_data 中未找到，已跳过。")
            continue

        traj_info = trajectory_to_episodes[episode_id]
        scan = traj_info[0]['scan']
        scene_dataset = scan.split('_')[0]
        scene = scan
        trajectory_id = str(traj_info[0]['trajectory_id'])

        group_key = (scene_dataset, scene, trajectory_id)
        if group_key not in trajectory_groups:
            trajectory_groups[group_key] = []
        trajectory_groups[group_key].append(lmdb_key)

    logger.info(f"分组完成，共 {len(trajectory_groups)} 个 trajectory。")

    # 清理旧目录
    repo_root = LEROBOT_HOME / repo_name
    if repo_root.exists():
        logger.warning(f"目标目录 {repo_root} 已存在，正在删除...")
        try:
            shutil.rmtree(repo_root)
            logger.info("已删除旧目录")
        except Exception as e:
            logger.error(f"删除旧目录失败: {e}")
            return
    LEROBOT_HOME.mkdir(parents=True, exist_ok=True)

    # 打开LMDB环境
    logger.info(f"正在打开LMDB环境: {lmdb_path}")
    lmdb_env = lmdb.open(str(lmdb_path_obj), readonly=True, lock=False, max_dbs=1)
    logger.info("LMDB环境已打开。")

    processed_count = 0
    failed_count = 0
    
    try:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {
                executor.submit(
                    process_trajectory_group,
                    lmdb_env=lmdb_env,
                    group_key=group_key,
                    lmdb_keys_in_group=lmdb_keys_list,
                    episode_to_trajectory=episode_to_trajectory,
                    scene_to_trajectories=scene_to_trajectories,
                    trajectory_to_episodes=trajectory_to_episodes,
                    repo_name=repo_name,
                    push_to_hub=push_to_hub,
                ): group_key
                for group_key, lmdb_keys_list in trajectory_groups.items()
            }
            
            progress_bar = tqdm.tqdm(as_completed(futures), total=len(futures), desc="转换 Trajectories")
            for future in progress_bar:
                group_key = futures[future]
                try:
                    _, success, message = future.result()
                    if success:
                        processed_count += 1
                    else:
                        failed_count += 1
                        logger.warning(f"处理失败: {message}")
                except Exception as e:
                    failed_count += 1
                    logger.error(f"处理 {group_key} 时出错: {e}")
                progress_bar.set_postfix_str(f"成功: {processed_count}, 失败: {failed_count}")
    finally:
        lmdb_env.close()
        logger.info("LMDB环境已关闭。")

    logger.info("="*50)
    logger.info("所有任务处理完成。")
    logger.info(f"成功处理: {processed_count} 个 trajectories")
    logger.info(f"处理失败: {failed_count} 个 trajectories")
    logger.info("="*50)

if __name__ == "__main__":
    main(
        lmdb_path="/cpfs/shared/simulation/vln-pe/20250314_sample_grutopia_vln10_controller/sample_data.lmdb",
        gather_data_json="data/vln_pe/raw_data/gruvln10/grutopia10/train/train.json.gz",
        repo_name="lerobot",
        # end_index=2,  # 测试前2个
    )

