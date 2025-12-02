#!/usr/bin/env python
"""
LMDB导航数据格式转换脚本 - 将LMDB中的导航数据转换为LeRobot v2.1格式
"""
import json
import hashlib
import shutil
import os
import tqdm
import numpy as np
import datasets
import logging
import threading
import sys
import datetime
from pathlib import Path
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterator, Dict, Any, List, Tuple, Generator
from datasets import concatenate_datasets
from lerobot.datasets.utils import embed_images, hf_transform_to_torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.utils import (
    # check_timestamps_sync,
    # get_episode_data_index,
    validate_episode_buffer,
    validate_frame,
    write_episodes,
    write_info,
    write_stats,
)
from lerobot.datasets.compute_stats import get_feature_stats, sample_indices, auto_downsample_height_width
from lerobot.datasets.video_utils import get_safe_default_codec
import lmdb
import msgpack_numpy
from PIL import Image
import torch
import torchvision
from collections import defaultdict
import gzip

# --- 全局变量定义 ---
LEROBOT_HOME = Path(os.environ.get("LEROBOT_HOME", "/shared/smartbot/vln-pe/gr10_data/grutopia10/lerobot_data"))
STRING_ENCODING_LENGTH = 64

# --- 预定义基础 Features ---
def create_base_features():
    """创建基础特征字典，不包含需要动态调整的特征"""
    return {
        # 图像特征 (存储为图像而非视频)
        # "observation.rgb": {
        #     "dtype": "int64",
        #     "shape": (256, 256, 3),
        #     "names": ["height", "width", "channel"]
        # },
        # "observation.depth": {
        #     "dtype": "float32",
        #     "shape": (256, 256),
        #     "names": ["height", "width"]
        # },

        # Camera Info - pano_camera_0
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

        # 动作特征 - 修复action路径
        "observation.action": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["action"]
        },
    }

def get_feature_shape(data, expected_dtype=None, default_shape=(1,)):
    """
    计算特征的形状
    
    参数:
        data: 输入数据
        expected_dtype: 预期的数据类型
        default_shape: 如果无法确定形状时使用的默认形状
        
    返回:
        特征的形状元组
    """
    # 转换数据为numpy数组
    data_array = ensure_numpy_array(data, expected_dtype, expand_dims=True)
    
    # 如果数据无效，返回默认形状
    if data_array is None:
        logger.warning(f"无法确定特征形状，使用默认形状 {default_shape}")
        return default_shape
        
    # 返回完整形状
    return data_array.shape

def ensure_numpy_array(data, expected_dtype=None, expand_dims=False):
    """确保数据是numpy数组，如果不是则进行转换"""
    if data is None:
        return None
        
    # 转换为numpy数组
    if not isinstance(data, np.ndarray):
        try:
            # 特别处理列表，确保正确转换为numpy数组
            if isinstance(data, list):
                # 检查列表是否包含标量或子列表
                if data and isinstance(data[0], list):
                    # 处理二维列表
                    data = np.array(data, dtype=expected_dtype)
                else:
                    # 处理一维列表
                    data = np.array(data, dtype=expected_dtype)
            else:
                # 处理其他类型
                data = np.array(data, dtype=expected_dtype)
        except Exception as e:
            logger.warning(f"无法将数据转换为numpy数组: {e}")
            return None
    
    # 如果需要，扩展维度
    if expand_dims and data.ndim == 0:
        data = np.expand_dims(data, axis=0)
        
    # 确保数据类型正确
    if expected_dtype is not None and data.dtype != expected_dtype:
        try:
            data = data.astype(expected_dtype)
        except Exception as e:
            logger.warning(f"无法将数据类型转换为 {expected_dtype}: {e}")
            
    return data

def compute_episode_stats(episode_buffer: dict, features: dict) -> dict:
    """计算episode的统计信息，对图像/视频数据进行采样以提高稳健性"""
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
                # 对于图像，data_list可能包含文件路径字符串，跳过统计
                if feature_info["dtype"] == "image" and data_list and isinstance(data_list[0], str):
                     continue
                data_array = np.stack(data_list)
            else:
                data_array = np.asarray(data_list)
        except (ValueError, np.exceptions.AxisError) as e:
            logger.warning(f"无法将特征 {key} 转换为numpy数组进行统计计算: {e}")
            continue

        try:
            if feature_info["dtype"] in ["image"]:  # 只处理图像，不处理视频
                sampled_indices = sample_indices(len(data_array))
                if len(sampled_indices) == 0:
                    logger.warning(f"特征 {key} 采样后无数据，跳过统计。")
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
            logger.warning(f"计算特征 {key} 的统计信息失败: {e}", exc_info=True)
            continue
    return ep_stats


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """将嵌套字典扁平化"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# --- 自定义 LeRobotDataset 子类 ---
class NavDatasetMetadata(LeRobotDatasetMetadata):
    def save_episode(
        self,
        episode_index: int,
        episode_length: int,
        episode_tasks: list[str],
        episode_stats: dict[str, dict],
    ) -> None:
        """扩展基类的save_episode方法，添加额外的元数据"""
        # 准备 episode 字典，包含必要的元数据
        episode_dict = {
            "episode_index": episode_index,
            "tasks": episode_tasks,
            "length": episode_length,
        }
        
        # 添加统计信息（扁平化）
        episode_dict.update(flatten_dict({"stats": episode_stats}))
        
        # 调用基类的 _save_episode_metadata 方法来处理缓冲和写入
        self._save_episode_metadata(episode_dict)

        # 更新 info
        self.info["total_episodes"] += 1
        self.info["total_frames"] += episode_length
        
        # 计算当前episode所属的chunk，并更新total_chunks
        chunk = episode_index // self.chunks_size
        if "total_chunks" not in self.info:
            self.info["total_chunks"] = 0
        if chunk >= self.info["total_chunks"]:
            self.info["total_chunks"] = chunk + 1

        self.info["splits"] = {"train": f"0:{self.info['total_episodes']}"}
        
        # 如果使用视频，更新视频信息
        if len(self.video_keys) > 0:
            self.update_video_info()

        write_info(self.info, self.root)

        # 更新统计信息（聚合所有 episode 的统计信息）
        self.stats = aggregate_stats([self.stats, episode_stats]) if self.stats else episode_stats
        write_stats(self.stats, self.root)

class NavDataset(LeRobotDataset):
    """自定义导航数据集类"""

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        features: dict,
        root: str | Path | None = None,
        robot_type: str | None = None,
        use_videos: bool = False,  # 禁用视频生成
        tolerance_s: float = 1e-4,
        image_writer_processes: int = 0,
        image_writer_threads: int = 4,
        video_backend: str | None = None,
    ) -> "NavDataset":
        """创建导航数据集，使用自定义的NavDatasetMetadata"""
        obj = cls.__new__(cls)
        obj.meta = NavDatasetMetadata.create(
            repo_id=repo_id,
            fps=fps,
            robot_type=robot_type,
            features=features,
            root=root,
            use_videos=use_videos,  # 禁用视频生成
        )
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
        """添加帧数据到episode_buffer，确保所有数据都是numpy数组"""
        # 确保所有torch张量转换为numpy数组
        for name in frame:
            if isinstance(frame[name], torch.Tensor):
                frame[name] = frame[name].numpy()
            # 确保列表转换为numpy数组
            elif isinstance(frame[name], list):
                frame[name] = np.array(frame[name])

        # 过滤掉图像特征进行验证
        features = {key: value for key, value in self.features.items() if key in self.hf_features}
        if 'task' not in frame:
            frame['task'] = {
                "dtype": "string",
                "shape": (1,),
                "names": ["navigation"]
            }
        validate_frame(frame, features)

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        # 自动添加frame_index和timestamp到episode buffer
        frame_index = self.episode_buffer["size"]
        if timestamp is None:
            timestamp = frame_index / self.fps
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)
        self.episode_buffer["task"].append(task)

        # 添加帧特征到episode_buffer
        for key, value in frame.items():
            # if key not in self.features:
            #     raise ValueError(
            #         f"帧中的元素不在特征列表中。'{key}' 不在 '{self.features.keys()}' 中。"
            #     )

            self.episode_buffer[key].append(value)

        self.episode_buffer["size"] += 1

    def save_episode(self, episode_instructions: List[Dict], trajectory_meta_info: List[Dict] = None) -> None:
        """保存episode数据，包括parquet文件"""
        if not self.episode_buffer:
            return

        episode_buffer = self.episode_buffer
        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        # 提取episode信息
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        new_tasks = []
        for task in tasks:
            if not isinstance(task, dict):
                new_tasks.append(task)
        tasks = new_tasks
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]  # 所有帧属于同一episode

        # 准备索引数据
        episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        # 处理任务索引
        try:
            for task in episode_tasks:
                task_index = self.meta.get_task_index(task)
                if task_index is None:
                    self.meta.add_task(task)
            episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])
        except Exception as e:
            episode_buffer["task_index"] = np.array([1 for i in range(len(tasks))])

        # 确保所有非图像特征都是正确形状的numpy数组
        for key, ft in self.features.items():
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["image"]:
                continue
                
            # 堆叠列表为numpy数组
            if isinstance(episode_buffer[key], list):
                # 确保列表中的每个元素都是numpy数组
                for i in range(len(episode_buffer[key])):
                    if not isinstance(episode_buffer[key][i], np.ndarray):
                        episode_buffer[key][i] = np.array(episode_buffer[key][i], dtype=ft["dtype"])
                
                # 堆叠成一个数组
                try:
                    episode_buffer[key] = np.stack(episode_buffer[key])
                except ValueError as e:
                    logger.warning(f"堆叠特征 {key} 时出错: {e}")
                    # 尝试用零填充不匹配的形状
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
        
        # 保存episode表格数据
        self._save_episode_table(episode_buffer, episode_index)
        
        # 保存元数据
        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats)

        # 检查时间戳同步
        # ep_data_index = get_episode_data_index(self.meta.episodes, [episode_index])
        # ep_data_index_np = {k: t.numpy() for k, t in ep_data_index.items()}
        # check_timestamps_sync(
        #     episode_buffer["timestamp"],
        #     episode_buffer["episode_index"],
        #     ep_data_index_np,
        #     self.fps,
        #     self.tolerance_s,
        # )

        # 保存自定义元数据
        episodes_file = self.root / "meta" / "episodes.jsonl"
        episodes_file.parent.mkdir(parents=True, exist_ok=True)
        with open(episodes_file, 'w', encoding='utf-8') as f:
            for instruction_dict in episode_instructions:
                f.write(json.dumps(instruction_dict, ensure_ascii=False) + "\n")
        logger.debug(f"Episodes (指令) 已保存至: {episodes_file}")

        if trajectory_meta_info:
            meta_info_file = self.root / "meta" / "trajectory_meta.jsonl"
            with open(meta_info_file, 'w', encoding='utf-8') as f:
                for meta_dict in trajectory_meta_info:
                    f.write(json.dumps(meta_dict, ensure_ascii=False) + "\n")
            logger.debug(f"Trajectory 元信息已保存至: {meta_info_file}")

        # 重置buffer
        self.episode_buffer = self.create_episode_buffer()

    def _save_episode_table(self, episode_buffer: dict, episode_index: int) -> None:
        """保存episode数据到parquet文件"""
        episode_dict = {key: episode_buffer[key] for key in self.hf_features}
        ep_dataset = datasets.Dataset.from_dict(episode_dict, features=self.hf_features, split="train")
        ep_dataset = embed_images(ep_dataset)
        self.hf_dataset = concatenate_datasets([self.hf_dataset, ep_dataset])
        self.hf_dataset.set_transform(hf_transform_to_torch)
        # ep_data_path = self.root / self.meta.get_data_file_path(ep_index=episode_index)
        
        # 直接计算chunk和file索引，避免调用get_data_file_path（它会尝试加载不存在的episodes）
        chunk_idx = episode_index // self.meta.chunks_size
        file_idx = episode_index % self.meta.chunks_size
        ep_data_path = self.root / self.meta.data_path.format(chunk_index=chunk_idx, file_index=file_idx)
        ep_data_path.parent.mkdir(parents=True, exist_ok=True)
        ep_dataset.to_parquet(ep_data_path)

def load_keys_from_file(key_path: Path) -> List[str]:
    """从文件加载key列表"""
    try:
        with open(key_path, 'r', encoding='utf-8-sig') as f:
            content = f.read().strip()
        if not content:
            return []
        if content.startswith('[') and content.endswith(']'):
            try:
                keys = json.loads(content)
                return [str(k) for k in keys]
            except json.JSONDecodeError:
                pass
        try:
            import ast
            parsed_content = ast.literal_eval(content)
            if isinstance(parsed_content, list):
                return [str(k) for k in parsed_content]
        except (ValueError, SyntaxError):
            pass
        raw_parts = []
        for line in content.splitlines():
             line = line.strip()
             if line:
                 if ',' in line:
                     raw_parts.extend(part.strip().strip('"\'') for part in line.split(','))
                 else:
                     raw_parts.extend(part.strip().strip('"\'') for part in line.split())
        keys = [part for part in raw_parts if part]
        return keys if keys else []
    except Exception as e:
        logger.error(f"加载key文件失败 {key_path}: {e}")
        raise

def load_gather_data(gather_data_json: Path) -> Tuple[Dict, Dict]:
    """加载gather_data.json并建立映射"""
    try:
        with open(gather_data_json, 'r') as f:
            data = json.load(f)
        episode_to_trajectory = {}
        scene_to_trajectories = {}
        trajectory_to_episodes = defaultdict(list)
        for scene_id, trajectories in data.items():
            scene_to_trajectories[scene_id] = []
            for traj_info in trajectories:
                episode_id = str(traj_info['episode_id'])
                trajectory_id = traj_info['trajectory_id']
                episode_to_trajectory[episode_id] = traj_info
                scene_to_trajectories[scene_id].append(traj_info)
                trajectory_to_episodes[trajectory_id].append(traj_info)
        logger.info(f"成功加载 gather_data.json，包含 {len(episode_to_trajectory)} 个 episodes.")
        return episode_to_trajectory, scene_to_trajectories, trajectory_to_episodes
    except Exception as e:
        logger.error(f"加载 gather_data.json 失败 {gather_data_json}: {e}")
        raise

def load_gather_data_gruvln10(gather_data_json: Path) -> Tuple[Dict, Dict]:
    """加载gather_data.json并建立映射"""
    try:
        with open(gather_data_json, 'r') as f:
            data = json.load(f)
        episode_to_trajectory = {}
        scene_to_trajectories = {}
        trajectory_to_episodes = defaultdict(list)
        for scene_id, trajectories in data.items():
            scene_to_trajectories[scene_id] = []
            for traj_info in trajectories:
                episode_id = str(traj_info['episode_id'])
                trajectory_id = traj_info['trajectory_id']
                episode_to_trajectory[str(trajectory_id)] = traj_info
                scene_to_trajectories[scene_id].append(traj_info)
                trajectory_to_episodes[trajectory_id].append(traj_info)
        logger.info(f"成功加载 gather_data.json，包含 {len(episode_to_trajectory)} 个 episodes.")
        return episode_to_trajectory, scene_to_trajectories, trajectory_to_episodes
    except Exception as e:
        logger.error(f"加载 gather_data.json 失败 {gather_data_json}: {e}")
        raise

def load_gather_data_gruvln10_new(data_json: Path) -> Tuple[Dict, Dict]:
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
    group_key: Tuple[str, str, int], # (scene_dataset, scene, trajectory_id)
    lmdb_keys_in_group: List[str],   # 该组包含的所有 LMDB keys
    episode_to_trajectory: Dict,
    scene_to_trajectories: Dict,
    repo_name: str,
    push_to_hub: bool,
):
    """处理一个完整的 trajectory group"""
    thread_name = threading.current_thread().name
    scene_dataset, scene, trajectory_id = group_key
    logger.info(f"线程 [{thread_name}] 开始处理 trajectory group: {scene_dataset}/{scene}/{trajectory_id} (包含 {len(lmdb_keys_in_group)} 个 LMDB keys)")

    # 1. 确定输出路径
    output_path = LEROBOT_HOME / repo_name / scene_dataset / scene / str(trajectory_id)

    # 2. 清理并创建目录
    if output_path.exists():
        logger.debug(f"线程 [{thread_name}] 清理残留目录: {output_path}")
        try:
            shutil.rmtree(output_path)
        except Exception as e:
            logger.error(f"线程 [{thread_name}] 清理目录失败: {e}")
            return (group_key, False, f"清理目录失败: {str(e)}")
    
    # output_path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"线程 [{thread_name}] 已创建输出目录: {output_path}")

    # 3. 选择一个 representative LMDB key
    representative_key = lmdb_keys_in_group[0]
    logger.debug(f"线程 [{thread_name}] 使用 key {representative_key} 作为 representative data source.")

    # 4. 从 representative key 读取数据
    with lmdb_env.begin() as txn:
        value_packed = txn.get(representative_key.encode())
        if value_packed is None:
            message = f"Representative key {representative_key} 在LMDB中未找到"
            logger.error(f"线程 [{thread_name}] {message}")
            return (group_key, False, message)
        value = msgpack_numpy.unpackb(value_packed)

    if 'episode_data' not in value:
        message = f"Representative key {representative_key} 的value中不包含 'episode_data'"
        logger.error(f"线程 [{thread_name}] {message}")
        return (group_key, False, message)

    episode_data = value['episode_data']
    finish_status_rep = value.get('finish_status', 'unknown')
    fail_reason_rep = value.get('fail_reason', 'unknown')

    # 将字符串编码为float32数组
    # finish_status_encoded = string_to_float32_array(finish_status_rep)
    # fail_reason_encoded = string_to_float32_array(fail_reason_rep)

    # 5. 提取和处理数据 (为 LeRobot 准备帧数据)
    lerobot_data_frames = []  # 存储每一帧的数据字典
    
    # --- 处理 RGB 帧 ---
    # rgb_frames = []
    # if 'camera_info' in episode_data and 'pano_camera_0' in episode_data['camera_info']:
    #     raw_rgb_data = episode_data['camera_info']['pano_camera_0'].get('rgb', None)
    #     raw_rgb_data = ensure_numpy_array(raw_rgb_data)
        
    #     if raw_rgb_data is not None and raw_rgb_data.size > 0:
    #         try:
    #             if raw_rgb_data.ndim == 4:  # (T, H, W, C)
    #                 rgb_data_list = [raw_rgb_data[i] for i in range(raw_rgb_data.shape[0])]
    #             elif raw_rgb_data.ndim == 3:  # (H, W, C)
    #                 rgb_data_list = [raw_rgb_data]
    #             else:
    #                 logger.warning(f"RGB 数据维度 {raw_rgb_data.ndim} 不符合预期。")
    #                 rgb_data_list = []

    #             if rgb_data_list:
    #                 for i, frame_array in enumerate(rgb_data_list):
    #                     if isinstance(frame_array, np.ndarray):
    #                         rgb_frames.append(frame_array.astype(np.int64))
    #                         # # 确保是 uint8
    #                         # if frame_array.dtype != np.uint8:
    #                         #     if np.issubdtype(frame_array.dtype, np.floating):
    #                         #         if frame_array.max() <= 1.0 and frame_array.min() >= 0.0:
    #                         #             frame_array_uint8 = (frame_array * 255).astype(np.uint8)
    #                         #         else:
    #                         #             frame_min, frame_max = frame_array.min(), frame_array.max()
    #                         #             if frame_max > frame_min:
    #                         #                 frame_array_uint8 = ((frame_array - frame_min) / (frame_max - frame_min) * 255).astype(np.uint8)
    #                         #             else:
    #                         #                 frame_array_uint8 = np.zeros_like(frame_array, dtype=np.uint8)
    #                         #     else:
    #                         #         frame_array_uint8 = frame_array.astype(np.uint8)
    #                         # else:
    #                         #     frame_array_uint8 = frame_array

    #                         # # 确保是 3 通道 (H, W, 3)
    #                         # if len(frame_array_uint8.shape) == 3 and frame_array_uint8.shape[2] >= 3:
    #                         #     if frame_array_uint8.shape[2] == 3:
    #                         #         rgb_frame = frame_array_uint8
    #                         #     elif frame_array_uint8.shape[2] == 4:
    #                         #         rgb_frame = frame_array_uint8[:, :, :3]
    #                         #     else:
    #                         #         logger.warning(f"RGB帧 {i} 通道数异常 ({frame_array_uint8.shape[2]})，取前3个通道。")
    #                         #         rgb_frame = frame_array_uint8[:, :, :3]
    #                         #     rgb_frames.append(rgb_frame)
    #                         # elif len(frame_array_uint8.shape) == 2:
    #                         #     logger.info(f"RGB帧 {i} 是灰度图，转换为3通道。")
    #                         #     rgb_frame = cv2.cvtColor(frame_array_uint8, cv2.COLOR_GRAY2RGB)
    #                         #     rgb_frames.append(rgb_frame)
    #                     else:
    #                         logger.warning(f"RGB帧列表中第 {i} 个元素不是 numpy 数组。")
    #             else:
    #                 logger.info(f"Representative key {representative_key}: RGB 数据解包后为空。")
    #         except Exception as e:
    #             logger.error(f"处理 representative key {representative_key} 的 RGB 数据时发生异常: {e}", exc_info=True)
    #     else:
    #         logger.info(f"Representative key {representative_key}: RGB 数据不存在、为空或不是 numpy 数组。")

    # # --- 处理 Depth 帧 ---
    # depth_frames = []
    # if 'camera_info' in episode_data and 'pano_camera_0' in episode_data['camera_info']:
    #     raw_depth_data = episode_data['camera_info']['pano_camera_0'].get('depth', None)
    #     raw_depth_data = ensure_numpy_array(raw_depth_data)
        
    #     if raw_depth_data is not None and raw_depth_data.size > 0:
    #         try:
    #             if raw_depth_data.ndim == 3:  # (T, H, W)
    #                 depth_data_list = [raw_depth_data[i] for i in range(raw_depth_data.shape[0])]
    #             elif raw_depth_data.ndim == 2:  # (H, W)
    #                 depth_data_list = [raw_depth_data]
    #             else:
    #                 logger.warning(f"Depth 数据维度 {raw_depth_data.ndim} 不符合预期。")
    #                 depth_data_list = []

                
    #             if depth_data_list:
    #                 depth_frames = depth_data_list
    #                 # for i, depth_array in enumerate(depth_data_list):
    #                 #     if isinstance(depth_array, np.ndarray) and depth_array.ndim == 2:
    #                 #         # 归一化深度图为 0-255 uint8 用于可视化
    #                 #         if depth_array.dtype != np.uint8:
    #                 #             if np.issubdtype(depth_array.dtype, np.floating):
    #                 #                 depth_min, depth_max = depth_array.min(), depth_array.max()
    #                 #                 if depth_max > depth_min:
    #                 #                     depth_norm = ((depth_array - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
    #                 #                 else:
    #                 #                     depth_norm = np.zeros_like(depth_array, dtype=np.uint8)
    #                 #                     if depth_max > 0:
    #                 #                         depth_norm.fill(255)
    #                 #             elif np.issubdtype(depth_array.dtype, np.integer):
    #                 #                 depth_min, depth_max = depth_array.min(), depth_array.max()
    #                 #                 if depth_max > depth_min:
    #                 #                     depth_norm = ((depth_array.astype(np.float64) - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
    #                 #                 else:
    #                 #                     depth_norm = np.zeros_like(depth_array, dtype=np.uint8)
    #                 #                     if depth_max > 0:
    #                 #                         depth_norm.fill(255)
    #                 #             else:
    #                 #                 depth_norm = depth_array.astype(np.uint8)
    #                 #         else:
    #                 #             depth_norm = depth_array
    #                 #         # 转换为 3 通道 BGR 图像
    #                 #         depth_bgr = np.stack([depth_norm, depth_norm, depth_norm], axis=-1)
    #                 #         depth_frames.append(depth_bgr)
    #             else:
    #                 logger.info(f"Representative key {representative_key}: Depth 数据解包后为空。")
    #         except Exception as e:
    #             logger.error(f"处理 representative key {representative_key} 的 Depth 数据时发生异常: {e}", exc_info=True)
    #     else:
    #         logger.info(f"Representative key {representative_key}: Depth 数据不存在、为空或不是 numpy 数组。")

    # 6. 处理非图像 episode_data
    flattened_episode_data = flatten_dict(episode_data, parent_key='observation')
    
    # 7. 提取动态特征并确定其形状
    
    key_mapping = {
        'observation.camera_info.pano_camera_0.rgb':'observation.rgb',
        'observation.camera_info.pano_camera_0.depth':'observation.depth',
        'observation.camera_info.pano_camera_0.position':'observation.camera_position',
        'observation.camera_info.pano_camera_0.orientation':'observation.camera_orientation',
        'observation.camera_info.pano_camera_0.yaw':'observation.camera_yaw',
        'observation.robot_info.position':'observation.robot_position',
        'observation.robot_info.orientation':'observation.robot_orientation',
        'observation.robot_info.yaw':'observation.robot_yaw',
        'observation.progress':'observation.progress',
        'observation.step':'observation.step',
        'observation.action':'observation.action',
    }
    new_flattened_episode_data={}
    for k,v in flattened_episode_data.items():
        if k in key_mapping:
            new_flattened_episode_data[key_mapping[k]]=v
        else:
            new_flattened_episode_data[k]=v
    flattened_episode_data = new_flattened_episode_data
    
    # 创建包含动态特征的特征字典
    features = create_base_features()

    # 8. 处理动作数据 - 修复action提取逻辑
    # 已知action与camera_info同级，都是episode_data的key
    action_data = episode_data.get('action', None)  # 直接从episode_data获取action
    
    # 确保action被正确提取并转换
    lerobot_data_buffer_non_video = {}

    # 如果没有有效的action数据，创建默认值
    if action_data is None:
        # 确定帧计数
        frame_count = 0
        # if rgb_frames:
        #     frame_count = len(rgb_frames)
        # elif depth_frames:
        #     frame_count = len(depth_frames)
        # else:
            # 从其他特征推断帧计数
        for key in ['observation.camera_info.pano_camera_0.yaw', 'observation.robot_info.yaw', 
                    'observation.progress', 'observation.step']:
            if key in flattened_episode_data:
                data = ensure_numpy_array(flattened_episode_data[key])
                # print(f"[{key}][{type(flattened_episode_data[key])}][{type(data)}]")
                if data is not None and data.ndim > 0:
                    frame_count = max(frame_count, data.shape[0])
        
        # 如果仍无法确定帧计数，使用默认值
        if frame_count == 0:
            frame_count = 1  # 至少有一帧
        
        # 创建默认action数组
        lerobot_data_buffer_non_video['observation.action'] = np.zeros((frame_count, 1), dtype=np.float64)
        logger.debug(f"创建默认action数据，形状: {(frame_count, 1)}")

    # 处理每个特征，确保它们是numpy数组并具有正确的形状
    for flat_key, flat_data in flattened_episode_data.items():
        if flat_key in ['observation.rgb', 'observation.depth']:
            continue
        if flat_key == "observation.action":
            new_flat_data = []
            for action in flat_data:
                if isinstance(action,list):
                    new_flat_data.append(action[0])
                else:
                    new_flat_data.append(action)
            flat_data = new_flat_data
            
        # 根据特征定义确定预期的数据类型
        expected_dtype = None
        if flat_key in features:
            dtype_str = features[flat_key]['dtype']
            if dtype_str == 'float32':
                expected_dtype = np.float32
            elif dtype_str == 'float64':
                expected_dtype = np.float64
            elif dtype_str == 'int64':
                expected_dtype = np.int64
        
        # 确保数据是numpy数组
        data_array = ensure_numpy_array(flat_data, expected_dtype, expand_dims=True)
        # print(f"[{flat_key}][{type(flat_data)}][{type(data_array)}]")
        if data_array is not None:
            # 处理不同维度的数据
            if data_array.ndim == 1:
                # 对于标量特征，扩展为(T, 1)形状
                if 'shape' in features.get(flat_key, {}) and len(features[flat_key]['shape']) > 0:
                    target_shape = features[flat_key]['shape']
                    if len(target_shape)==1 and target_shape[0] == 1:
                        data_array = np.expand_dims(data_array, axis=-1)
                    else:
                        logger.warning(f"特征 {flat_key} 形状不匹配，预期 {target_shape}，实际 {data_array.shape}")
            
            # 存储处理后的数组
            lerobot_data_buffer_non_video[flat_key] = data_array

    # 9. 确定帧数
    frame_count = 0
    if lerobot_data_buffer_non_video:
        # 找到最大的维度作为帧数
        for v in lerobot_data_buffer_non_video.values():
            if isinstance(v, np.ndarray) and v.ndim > 0:
                current_count = v.shape[0]
                if current_count > frame_count:
                    frame_count = current_count
    
    # 如果从非视频数据中未找到帧数，则使用视频帧的数量
    # if frame_count == 0 and (rgb_frames or depth_frames):
    #     frame_count = max(len(rgb_frames), len(depth_frames))

    # 10. 广播状态信息（已编码为float32数组）
    # if frame_count > 0:
    #     # 确保状态信息具有正确的形状
    #     finish_status_broadcast = np.broadcast_to(
    #         finish_status_encoded, 
    #         (frame_count, STRING_ENCODING_LENGTH)
    #     )
    #     fail_reason_broadcast = np.broadcast_to(
    #         fail_reason_encoded, 
    #         (frame_count, STRING_ENCODING_LENGTH)
    #     )
        
        # lerobot_data_buffer_non_video['observation.finish_status'] = finish_status_broadcast
        # lerobot_data_buffer_non_video['observation.fail_reason'] = fail_reason_broadcast
    # else:
    #     logger.warning(f"Trajectory group {group_key}: 无法确定帧长度，未添加 finish_status 和 fail_reason。")

    # 11. 确保所有非图像特征的长度与帧计数匹配
    for key in list(lerobot_data_buffer_non_video.keys()):
        data = lerobot_data_buffer_non_video[key]
        if isinstance(data, np.ndarray) and data.ndim > 0:
            # 如果数据长度与帧计数不匹配，则广播或截断
            if data.shape[0] != frame_count:
                logger.warning(f"特征 {key} 长度不匹配，预期 {frame_count}，实际 {data.shape[0]}，正在调整...")
                
                # 获取特征的目标形状
                target_shape = features.get(key, {}).get('shape', ())
                feature_size = np.prod(target_shape) if target_shape else 1
                
                # 重塑数据以确保正确的特征维度
                if data.size % feature_size == 0:
                    data = data.reshape(-1, *target_shape)
                else:
                    logger.warning(f"特征 {key} 无法重塑为目标形状 {target_shape}")
                    continue
                
                # 调整帧数以匹配
                if data.shape[0] < frame_count:
                    # 重复数据以达到所需长度
                    repeat_factor = (frame_count // data.shape[0]) + 1
                    data = np.repeat(data, repeat_factor, axis=0)[:frame_count]
                else:
                    # 截断数据
                    data = data[:frame_count]
                
                lerobot_data_buffer_non_video[key] = data

    # 12. 组装每一帧的数据字典
    if frame_count > 0:
        for i in range(frame_count):
            frame_data = {}
            # 添加非图像数据
            # frame_data['task'] = 'navigation'
            for feat_key, feat_array in lerobot_data_buffer_non_video.items():
                if isinstance(feat_array, np.ndarray) and feat_array.ndim > 0 and i < feat_array.shape[0]:
                    # 提取第i帧的数据
                    frame_data[feat_key] = feat_array[i]
                else:
                    # 使用默认值填充
                    dummy_shape = features.get(feat_key, {}).get('shape', ())
                    dummy_dtype = features.get(feat_key, {}).get('dtype', 'float64')
                    frame_data[feat_key] = np.zeros(dummy_shape, dtype=dummy_dtype)
            
            # 添加图像数据到 observation.rgb 和 observation.depth
            # if i < len(rgb_frames):
            #     frame_data["observation.rgb"] = rgb_frames[i]
            # else:
            #     # 如果没有该帧的RGB数据，使用默认值
            #     frame_data["observation.rgb"] = np.zeros(
            #         features["observation.rgb"]["shape"], 
            #         dtype=np.uint8
            #     )
                
            # if i < len(depth_frames):
            #     frame_data["observation.depth"] = depth_frames[i]
            # else:
            #     # 如果没有该帧的深度数据，使用默认值
            #     frame_data["observation.depth"] = np.zeros(
            #         features["observation.depth"]["shape"], 
            #         dtype=np.uint8
            #     )

            lerobot_data_frames.append(frame_data)
    else:
        message = f"Trajectory group {group_key} 没有有效帧数据"
        logger.warning(f"线程 [{thread_name}] {message}")
        return (group_key, False, message)

    # 13. 创建数据集实例 - 使用包含动态特征的features
    dataset = NavDataset.create(
        repo_id=repo_name,
        root=output_path, 
        robot_type="unknown",
        fps=6, 
        use_videos=False,  # 禁用视频生成
        features=features,
    )

    # 14. 逐帧添加数据到 LeRobot Dataset
    task_text = f"Navigation task for trajectory {trajectory_id}"

    for i, frame_data in enumerate(lerobot_data_frames):
        dataset.add_frame(frame=frame_data, task=task_text, timestamp=i/6.0)

    # 15. 准备 meta/episodes.jsonl 的内容
    all_episodes_for_traj = [ep for ep in scene_to_trajectories.get(scene, []) if ep['trajectory_id'] == trajectory_id]
    all_episodes_for_traj.sort(key=lambda x: x['episode_id'])
    episode_instructions = []
    trajectory_meta_info = []
    for ep_info in all_episodes_for_traj:
        ep_id_str = str(ep_info['episode_id'])
        episode_instructions.append({
            "episode_id": ep_id_str,
            "instruction_text": ep_info['instruction']['instruction_text'],
            "finish_status": finish_status_rep,
            "fail_reason": fail_reason_rep,
        })
        if 'instruction_tokens' in ep_info['instruction']:
            episode_instructions[-1]['instruction_tokens'] = ep_info['instruction']['instruction_tokens']
        else:
            episode_instructions[-1]['instruction_tokens'] = []
    # 16. 保存episode 
    dataset.save_episode(
        episode_instructions=episode_instructions,
        trajectory_meta_info=trajectory_meta_info
    )

    dataset._wait_image_writer()

    # 保存numpy
    rgb = value['episode_data']['camera_info']['pano_camera_0']['rgb']
    rgb_path = os.path.join(output_path,"videos/chunk-000/observation.images.rgb")
    os.makedirs(rgb_path)
    np.save(os.path.join(rgb_path, 'rgb.npy'), rgb)

    depth = value['episode_data']['camera_info']['pano_camera_0']['depth']
    depth_path = os.path.join(output_path,"videos/chunk-000/observation.images.depth")
    os.makedirs(depth_path)
    np.save(os.path.join(depth_path, 'depth.npy'), depth)

    message = f"成功处理 trajectory group: {scene_dataset}/{scene}/{trajectory_id}, 包含 {len(lmdb_keys_in_group)} 个 keys, {frame_count} 帧"
    logger.info(f"线程 [{thread_name}] {message}")
    return (group_key, True, message)


def main(
    lmdb_path: str,
    # key_path: str,
    gather_data_json: str,
    repo_name: str = "lerobot",
    *,
    push_to_hub: bool = False,
    num_threads: int = 1, # 强制单线程以避免文件系统冲突
    start_index: int = 0,
    end_index: int | None = None,
):
    """LMDB到LeRobot转换主函数"""

    log_dir = Path("/cpfs/user/liuyu2/projects/logs/vln_pe")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    end_str = "end" if end_index is None else str(end_index)
    log_file_path = log_dir / f"s{start_index}-e{end_str}_{timestamp}.log"
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    logger.add(log_file_path, level="DEBUG", rotation="100 MB", encoding="utf-8")
    logger.info("="*50)
    logger.info(f"脚本启动，将处理索引范围: [{start_index}, {end_index if end_index is not None else 'end'})")
    logger.info(f"日志文件位于: {log_file_path}")
    logger.info(f"将使用 {num_threads} 个线程进行处理。")
    logger.info("="*50)

    lmdb_path_obj = Path(lmdb_path)
    # key_path_obj = Path(key_path)
    gather_data_json_obj = Path(gather_data_json)

    logger.info(f"正在加载key列表...")
    # all_keys = load_keys_from_file(key_path_obj)
    all_keys = []
    with lmdb.open(
        lmdb_path,
        readonly=True,
        lock=False,
    ) as lmdb_env:
        # Obtain all keys
        with lmdb_env.begin() as txn:
            cursor = txn.cursor()
            while cursor.next():
                all_keys.append(cursor.key().decode())
    print(all_keys)
    if not all_keys:
        logger.warning("没有找到任何key。退出。")
        return
    logger.info(f"加载了 {len(all_keys)} 个key。")

    logger.info(f"正在加载 gather_data.json...")
    # episode_to_trajectory, scene_to_trajectories, trajectory_to_episodes = load_gather_data(gather_data_json_obj) # !!!
    episode_to_trajectory, scene_to_trajectories, trajectory_to_episodes = load_gather_data_gruvln10_new(gather_data_json_obj) # !!!
    logger.info(f"gather_data.json 加载完成。")

    selected_keys = all_keys[start_index:end_index]
    if not selected_keys:
        logger.warning("在指定的索引范围内没有找到需要处理的key。退出。")
        return
    logger.info(f"选定 {len(selected_keys)} 个key进行处理。")

    logger.info("正在对选定的 keys 进行分组...")
    trajectory_groups = {}

    for lmdb_key in selected_keys:
        episode_id = lmdb_key # here I use trajectory_id as the key in lmdb
        if '_' in episode_id:  # 判断字符串中是否有下划线
            episode_id = episode_id.split('_')[0]
        if episode_id not in trajectory_to_episodes:
            logger.warning(f"Key {lmdb_key} 在 gather_data.json 中未找到，已跳过。")
            continue

        traj_info = trajectory_to_episodes[episode_id]
        scan = traj_info[0]['scan']
        # scene_parts = scene_id_full_path.split('/')
        scene_dataset = scan.split('_')[0]
        scene = scan
        trajectory_id = traj_info[0]['trajectory_id']

        group_key = (scene_dataset, scene, trajectory_id)
        if group_key not in trajectory_groups:
            trajectory_groups[group_key] = []
        trajectory_groups[group_key].append(lmdb_key)

    logger.info(f"分组完成，共 {len(trajectory_groups)} 个独立的 trajectory 需要处理。")

    repo_root = LEROBOT_HOME / repo_name
    if repo_root.exists():
        logger.warning(f"目标 repo 目录 {repo_root} 已存在，正在删除...")
        try:
            shutil.rmtree(repo_root)
            logger.info(f"已删除旧的 repo 目录 {repo_root}")
        except Exception as e:
            logger.error(f"删除旧的 repo 目录 {repo_root} 失败: {e}")
            return
    LEROBOT_HOME.mkdir(parents=True, exist_ok=True)

    logger.info(f"正在打开LMDB环境: {lmdb_path}")
    lmdb_env = lmdb.open(str(lmdb_path_obj), readonly=True, lock=False, max_dbs=1)
    logger.info("LMDB环境已打开。")

    processed_count = 0
    failed_count = 0
    try:
        effective_num_threads = 1
        logger.info(f"为避免并发冲突，实际使用 {effective_num_threads} 个线程处理 trajectory groups。")

        with ThreadPoolExecutor(max_workers=effective_num_threads) as executor:
            futures = {
                executor.submit(
                    process_trajectory_group,
                    lmdb_env=lmdb_env,
                    group_key=group_key,
                    lmdb_keys_in_group=lmdb_keys_list,
                    episode_to_trajectory=episode_to_trajectory,
                    scene_to_trajectories=scene_to_trajectories,
                    repo_name=repo_name,
                    push_to_hub=push_to_hub,
                ): group_key
                for group_key, lmdb_keys_list in trajectory_groups.items()
            }
            progress_bar = tqdm.tqdm(as_completed(futures), total=len(futures), desc="转换 Trajectories")
            for future in progress_bar:
                group_key = futures[future]
                _, success, message = future.result()
                if success:
                    processed_count += 1
                else:
                    failed_count += 1
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
    # 用于调试，正式运行时使用tyro.cli(main)
    # tyro.cli(main)
    # main(
    #     lmdb_path="/shared/smartbot/vln-pe/20250211_sample_origin/sample_data.lmdb",
    #     key_path="/shared/smartbot/vln-pe/test/keys.txt",
    #     gather_data_json="/shared/smartbot/vln-pe/data/datasets/R2R_VLNCE_v1-3_corrected/gather_data/train_gather_data.json",
    #     end_index=5,
    # )
    main(
        lmdb_path="/cpfs/shared/simulation/vln-pe/20250314_sample_grutopia_vln10_controller/sample_data.lmdb",
        # gather_data_json="/cpfs/user/wangliuyi/code/internnav_wangyukai/data/vln_pe/raw_data/gruvln10/grutopia10/gather_data/train_gather_data.json",
        # gather_data_json="data/vln_pe/raw_data/gruvln10/grutopia10/gather_data_seen0.8/train_gather_data.json",
        gather_data_json="data/vln_pe/raw_data/gruvln10/grutopia10/train/train.json.gz",
        # repo_name="test",
        # end_index=5,
    )
