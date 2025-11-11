''' For converting R2R dataset to R2R-CE dataset
Author: wangliuyi
Date: 2025-07-07
'''

import os
import json
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import gzip
from tqdm import tqdm
from decimal import Decimal
from collections import defaultdict
import copy
from scipy.spatial.transform import Rotation as R
from test.glove_embedding import InstructionEmbedding

def euler_angles_to_quat(angles, degrees=False):
    """
    Convert Euler angles (roll, pitch, yaw) to quaternion.

    Args:
        angles (list or np.array): Euler angles [roll, pitch, yaw] in degrees.

    Returns:
        np.array: Quaternion [w, x, y, z].
    """
    r = R.from_euler('xyz', angles, degrees=degrees)
    quat = r.as_quat()
    return [quat[3], quat[0], quat[1], quat[2]]

def quat_to_euler_angles(quat):
    """
    Convert quaternion to Euler angles (roll, pitch, yaw).

    Args:
        quat (list or np.array): Quaternion [w, x, y, z].

    Returns:
        np.array: Euler angles [roll, pitch, yaw] in degrees.
    """
    reordered_quat = [quat[1], quat[2], quat[3], quat[0]]
    r = R.from_quat(reordered_quat)
    angles = r.as_euler('xyz', degrees=True)
    return angles

def read_json_file(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def read_json_gz_file(path):
    with gzip.open(path, 'rb') as f:
        data = json.load(f)
    return data

def load_sixth_floor_data(dataset_base_dir, split):
    ''' Load data based on VLN-CE
    '''
    total_scans = []
    load_data = []
    dataset_file = os.path.join(dataset_base_dir, f"{split}", f"{split}.json.gz")
    with gzip.open(dataset_file, 'rt', encoding='utf-8') as f:
        data = json.load(f)
        for item in data["episodes"]:
            item["original_start_position"] = copy.copy(item["start_position"])
            item["original_start_rotation"] = copy.copy(item["start_rotation"])
            item["start_position"] = [item["original_start_position"][0], item["original_start_position"][1], item["original_start_position"][2]] # unchanged
            init_orientation = item['original_start_rotation']
            init_orientation = quat_to_euler_angles(init_orientation)
            orientation = [0, 0, init_orientation[2]]
            init_orientation = euler_angles_to_quat(orientation)
            item['start_rotation'] = init_orientation
            item["scan"] = item["scene_id"]
            item["c_reference_path"] = []
            if "reference_path" in item.keys():
                for path in item["reference_path"]:
                    item["c_reference_path"].append([path[0], path[1], path[2]])
                item["reference_path"] = item["c_reference_path"]
                del item["c_reference_path"]
            load_data.append(item)
            total_scans.append(item["scan"])
    print(f"Loaded data with a total of {len(load_data)} items from {split}")
    return load_data, list(set(total_scans))

def load_data(dataset_base_dir, split):
    ''' Load data based on VLN-CE
    '''
    total_scans = []
    load_data = []
    dataset_file = os.path.join(dataset_base_dir, f"{split}", f"{split}.json.gz")
    with gzip.open(dataset_file, 'rt', encoding='utf-8') as f:
        data = json.load(f)
        for idx, item in enumerate(data["episodes"]):
            item["original_start_position"] = copy.copy(item["start_position"])
            item["original_start_rotation"] = copy.copy(item["start_rotation"])
            item["start_position"] = [item["original_start_position"][0], -item["original_start_position"][2], item["original_start_position"][1]]
            item["start_rotation"] = [-item["original_start_rotation"][3], item["original_start_rotation"][0], item["original_start_rotation"][2], -item["original_start_rotation"][1]] # [x,y,z,-w] => [w,x,y,z]
            item["start_rotation"] = transform_rotation_z_90degrees(item["start_rotation"])
            if "/" in item["scene_id"]:
                item["scan"] = item["scene_id"].split("/")[1]
            else:
                item["scan"] = item["scene_id"]
            item["c_reference_path"] = []
            if "reference_path" in item.keys():
                for path in item["reference_path"]:
                    item["c_reference_path"].append([path[0], -path[2], path[1]])
                    # item["c_reference_path"].append([path[0], path[1], path[2]])
                item["reference_path"] = item["c_reference_path"]
                del item["c_reference_path"]
            load_data.append(item)
            total_scans.append(item["scan"])

    print(f"Loaded data with a total of {len(load_data)} items from {split}")
    return load_data, list(set(total_scans))

def transform_rotation_z_90degrees(rotation):
    ''' 沿着z轴旋转90度
    '''
    z_rot_90 = [np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)]  # 90 degrees = pi/2 radians
    w1, x1, y1, z1 = rotation
    w2, x2, y2, z2 = z_rot_90
    revised_rotation = [
        w1*w2 - x1*x2 - y1*y2 - z1*z2,  # w
        w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
        w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
        w1*z2 + x1*y2 - y1*x2 + z1*w2   # z
    ]
    return revised_rotation

def get_yaw_from_rotation(rotation):
    """从四元数计算yaw角(绕z轴的旋转)
    Args:
        rotation: 四元数 [w, x, y, z]
    Returns:
        yaw: 弧度制的偏航角
    """
    w, x, y, z = rotation
    # 计算yaw(绕z轴旋转)的弧度值
    yaw = np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    return yaw

# 如果需要转换为角度制:
def get_yaw_degree(rotation):
    """从四元数计算yaw角并转换为角度制
    """
    yaw_rad = get_yaw_from_rotation(rotation)
    yaw_deg = np.degrees(yaw_rad)
    return yaw_deg


class R2RDataToHabitat():
    def __init__(self):
        self.connectivity_file = "data/connectivity_mp3d/%s_connectivity.json"
        self.connectivity_dir = "data/connectivity_mp3d"
        
        # load connectivity
        self.connectivity_dict = self.read_all_connectivity(self.connectivity_dir)
        
        # init the glove embedding
        self.embedder = InstructionEmbedding(source_folder="data/datasets/R2R_VLNCE_v1-3_preprocessed")
    
    def convert(self, ori_data, split):
        new_data = []
        self.episode_id = 0
        for data in tqdm(ori_data):
            processed_data = self.convert_r2r_to_vlnce(self.connectivity_dict, data)
            if processed_data is not None:
                new_data.extend(processed_data)
            
        print(f"save {len(new_data)} episodes")
        output_path = R2R_test_dataset_path.replace('.json', '.json.gz')
        with gzip.open(output_path, 'wb') as f:
            write_data = {'episodes': new_data}
            f.write(json.dumps(write_data).encode('utf-8'))
        print(f"save {len(new_data)} episodes to {output_path}")
        
        # save gt path (for habitat eval)
        # ep_ids = {i:[] for i in range(len(new_data))}
        # gt_path = output_path.replace('.json.gz', '_gt.json.gz')
        # with gzip.open(gt_path, 'wb') as f:
        #     f.write(json.dumps(ep_ids, indent=4).encode('utf-8'))
        # print(f"save {len(new_data)} episodes to gt_path: {gt_path}")
        
    def yaw2quat(self, yaw, to_habitat=True):
        """将yaw角（绕z轴旋转的角度）转换为四元数
        Args:
            yaw: 绕z轴旋转的角度（弧度）
        Returns:
            quaternion: [w, x, y, z] 格式的四元数
        """
        if isinstance(yaw, Decimal):
            yaw = float(yaw)
        
        w = np.cos(yaw / 2.0)
        x = 0.0
        y = 0.0
        z = np.sin(yaw / 2.0)
        
        if to_habitat:
            quaternion = [x,z,y,-w]
        else:
            quaternion = [w, x, y, z]
        return quaternion

    def pos2habitat(self, pos):
        # return [pos[0], pos[2]-1.50, -pos[1]]
        return [pos[0], pos[2]-1.40, -pos[1]]
        # return [pos[0], pos[1], pos[2]]

    def read_all_connectivity(self, connectivity_path):
        """遍历 connectivity_dir 下的所有文件，按 scan 索引文件信息"""
        connectivity_dict = {}
        for filename in tqdm(os.listdir(connectivity_path)):
            if filename.endswith('_connectivity.json'):
                scan = filename.split('_')[0]  # 获取文件名中的 scan 部分
                file_path = os.path.join(connectivity_path, filename)
                connectivity_dict[scan] = read_json_file(file_path)  # 读取文件并存储在字典中
        print(f"read {len(connectivity_dict)} connectivity files")
        return connectivity_dict

    def convert_r2r_to_vlnce(self,connectivity_dict, data_item, scan_nums=None, max_nums_per_scan=None):
        new_data = []
        scan = data_item['scan']
        if scan_nums is not None:
            if scan_nums[scan] > max_nums_per_scan:
                return None
            else:
                scan_nums[scan] += 1
        
        heading = data_item['heading']
        path_connectivity = connectivity_dict[scan]
        conn_img2pos_dict = {}
        for conn in path_connectivity:
            conn_img2pos_dict[conn['image_id']] = [conn['pose'][3], conn['pose'][7], conn['pose'][11]] # [x, y, z]
        
        path_pos = []
        for path in data_item['path']:
            path_pos.append(self.pos2habitat(conn_img2pos_dict[path]))
        
        # save data the same as vlnce-habitat
        for idx in range(len(data_item['instructions'])):
            data_processed = {}
            data_processed['trajectory_id'] = data_item['path_id']
            data_processed['episode_id'] = self.episode_id
            self.episode_id += 1
            data_processed['scene_id'] = f'mp3d/{scan}/{scan}.glb' ### 这里要和scene_datasets的路径一致
            data_processed['start_position'] = path_pos[0]
            data_processed['start_rotation'] = self.yaw2quat(heading, to_habitat=True)
            data_processed['info'] = {'geodesic_distance': data_item['distance']} # meaningless
            data_processed['goals'] = [{'position': path_pos[-1], 'radius': 3.0}]
            tokens, vocab, embeddings = self.embedder.embedding(data_item['instructions'][idx])
            data_processed['instruction'] = {
                'instruction_text': data_item['instructions'][idx],
                'instruction_tokens': tokens
            }
            data_processed['reference_path'] = path_pos
            
            new_data.append(data_processed)
        
        return new_data

class datasetGather:
    def __init__(self, dataset_base_dir, is_sixth_floor_data=False):
        self.dataset_base_dir = dataset_base_dir
        # self.splits = ['train', 'val_seen', 'val_unseen']
        # self.splits = ['envdrop']
        # self.splits = ['train', 'val_seen']
        self.splits = ['test']
        self.data = {split: [] for split in self.splits}
        self.scan = {}
        for split in self.splits:
            if is_sixth_floor_data: 
                self.data[split], self.scan[split] = load_sixth_floor_data(self.dataset_base_dir, split)
            else:
                self.data[split], self.scan[split] = load_data(self.dataset_base_dir, split)

    def gatherSameScanData(self, save_gather_data=True, save_dir='gather_data/', fix_rotation=False):
        scan2data = {split: {} for split in self.splits}
        for split in self.splits:
            for data in self.data[split]:
                scan = data['scan']
                if scan not in scan2data[split]:
                    scan2data[split][scan] = []
                if fix_rotation:
                    # no need. already turn in load_data.
                    data['start_rotation'] = transform_rotation_z_90degrees(data['start_rotation'])
                scan2data[split][scan].append(data)
        
        if save_gather_data:
            target_save_dir = os.path.join(self.dataset_base_dir, 'gather_data')
            if not os.path.exists(target_save_dir):
                os.makedirs(target_save_dir)
            
            for split in self.splits:
                save_path = os.path.join(target_save_dir, f'{split}_gather_data.json')
                with open(save_path, 'w') as f:
                    json.dump(scan2data[split], f, indent=2)
                print(f'Saved gathered data for {split} to {save_path}')
            
            with open(os.path.join(target_save_dir, 'env_scan.json'), 'w') as f:
                json.dump(self.scan, f, indent=2)
            print(f'Saved scan data to {os.path.join(target_save_dir, "env_scan.json")}')
        
        return scan2data

if __name__ == "__main__":
    # R2R_dataset_dir = "data/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz"
    R2RCE_test_dataset_path = "data/datasets/R2R_VLNCE_v1-3_preprocessed/test/test.json.gz"
    R2R_test_dataset_dir = "data/datasets/R2R_VLNCE_test"
    R2R_test_dataset_path = "data/datasets/R2R_VLNCE_test/test/test.json"
    
    # 1. convert coordinate system (R2R -> R2R-CE)
    converter = R2RDataToHabitat()
    split = 'test'
    ori_data = read_json_file(R2R_test_dataset_path)
    converter.convert(ori_data, split) 
    
    # 2. gather data
    dataset_gather = datasetGather(R2R_test_dataset_dir)
    scan2data = dataset_gather.gatherSameScanData(save_gather_data=True, save_dir='gather_data/', fix_rotation=False)
        