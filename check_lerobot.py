import os
import lmdb
import msgpack_numpy
import pandas as pd
import numpy as np
import json
import gzip
# from lerobot.common.datasets.utils import embed_images

lerobot_path = "data/vln_pe/traj_data/gruvln10"
lmdb_path="/cpfs/shared/simulation/vln-pe/20250314_sample_grutopia_vln10_controller/sample_data.lmdb"
data_json_file="data/data_old/grutopia10/train/train.json.gz"

def load_dataset_and_gather(data_json_file):
    with gzip.open(data_json_file, 'rt', encoding='utf-8') as f:
        load_data = json.load(f)['episodes']
    
    new_data = {}
    episodeId_to_trajId = {}
    for episode in load_data:
        if episode['scan'] not in new_data:
            new_data[episode['scan']] = []
        new_data[episode['scan']].append(episode)
        episodeId_to_trajId[str(episode['episode_id'])] = str(episode['trajectory_id'])
    return new_data, episodeId_to_trajId

gather_data, episodeId_to_trajId = load_dataset_and_gather(data_json_file)

lmdb_env = lmdb.open(lmdb_path, readonly=True, lock=False, max_dbs=1)
# 查看所有 key
# with lmdb_env.begin() as txn:
#     cursor = txn.cursor()
#     all_keys = [key.decode() for key, _ in cursor]
#     print(f"LMDB 中共有 {len(all_keys)} 个 key")
        
a = 0

for scan in os.listdir(lerobot_path):
    
    scan_path = os.path.join(lerobot_path, scan)
    if not os.path.isdir(scan_path):
        continue
    for trajectory in os.listdir(scan_path):
        a +=1
        trajectory_path = os.path.join(scan_path, trajectory)
        if not os.path.isdir(trajectory_path):
            continue
        print(f"[{a}][scan:{scan}][episode:{trajectory}]")

        trajectory = episodeId_to_trajId[trajectory]
        # 获取老数据
        with lmdb_env.begin() as txn:
            # value_packed = txn.get(f"{trajectory}_{trajectory}".encode())
            value_packed = txn.get(f"{trajectory}".encode())
            if value_packed is None:
                print(f"trajectory :{trajectory} 在LMDB中未找到")
                continue
            value = msgpack_numpy.unpackb(value_packed)
        # 解析新数据
        df = pd.read_parquet(os.path.join(trajectory_path, "data/chunk-000/episode_000000.parquet"))
        lerobot_ = np.array(df['observation.camera_position'].tolist())
        lmdb_ = value['episode_data']['camera_info']['pano_camera_0']['position']
        assert type(lerobot_) == type(lmdb_)
        assert lerobot_.shape == lmdb_.shape
        for i in range(len(lerobot_)):
            assert (lerobot_[i] == lmdb_[i]).all()
        print(f"")
        # 'observation.camera_orientation', 
        lerobot_ = np.array(df['observation.camera_orientation'].tolist())
        lmdb_ = value['episode_data']['camera_info']['pano_camera_0']['orientation']
        assert type(lerobot_) == type(lmdb_)
        assert lerobot_.shape == lmdb_.shape
        for i in range(len(lerobot_)):
            assert (lerobot_[i] == lmdb_[i]).all()
        # 'observation.camera_yaw', 
        lerobot_ = np.array(df['observation.camera_yaw'].tolist())
        lmdb_ = value['episode_data']['camera_info']['pano_camera_0']['yaw']
        assert type(lerobot_) == type(lmdb_)
        assert lerobot_.shape == lmdb_.shape
        for i in range(len(lerobot_)):
            assert (lerobot_[i] == lmdb_[i]).all()
        # 'observation.robot_position', 
        lerobot_ = np.array(df['observation.robot_position'].tolist())
        lmdb_ = value['episode_data']['robot_info']['position']
        assert type(lerobot_) == type(lmdb_)
        assert lerobot_.shape == lmdb_.shape
        for i in range(len(lerobot_)):
            assert (lerobot_[i] == lmdb_[i]).all()
        # 'observation.robot_orientation', 
        lerobot_ = np.array(df['observation.robot_orientation'].tolist())
        lmdb_ = value['episode_data']['robot_info']['orientation']
        assert type(lerobot_) == type(lmdb_)
        assert lerobot_.shape == lmdb_.shape
        for i in range(len(lerobot_)):
            assert (lerobot_[i] == lmdb_[i]).all()
        # 'observation.robot_yaw', 
        lerobot_ = np.array(df['observation.robot_yaw'].tolist())
        lmdb_ = value['episode_data']['robot_info']['yaw']
        assert type(lerobot_) == type(lmdb_)
        assert lerobot_.shape == lmdb_.shape
        for i in range(len(lerobot_)):
            assert (lerobot_[i] == lmdb_[i]).all()
        # 'observation.progress', 
        lerobot_ = np.array(df['observation.progress'].tolist())
        lmdb_ = value['episode_data']['progress']
        assert type(lerobot_) == type(lmdb_)
        assert lerobot_.shape == lmdb_.shape
        for i in range(len(lerobot_)):
            assert (lerobot_[i] == lmdb_[i]).all()
        # 'observation.step', 
        lerobot_ = np.array(df['observation.step'].tolist())
        lmdb_ = value['episode_data']['step']
        assert type(lerobot_) == type(lmdb_)
        assert lerobot_.shape == lmdb_.shape
        for i in range(len(lerobot_)):
            assert (lerobot_[i] == lmdb_[i]).all()
        # 'observation.action', 
        lerobot_ = df['observation.action'].tolist()
        lmdb_ = value['episode_data']['action']
        if isinstance(lmdb_[-1], list):
            lmdb_[-1] = lmdb_[-1][0]
        assert type(lerobot_) == type(lmdb_)
        assert len(lerobot_) == len(lmdb_)
        for i in range(len(lerobot_)):
            assert lerobot_[i] == lmdb_[i]
        
        # 读取json 文件
        target_episodes = {}
        for episode in gather_data[scan]:
            if str(episode['trajectory_id']) == trajectory:
                target_episodes[str(episode['episode_id'])] = episode
        episodes_in_json = {}
        finish_status_in_json = None
        fail_reason_in_json = None
        with open(os.path.join(trajectory_path, "meta/episodes.jsonl"), 'r') as f:
            for line in f:
                try:
                    json_data = json.loads(line.strip())  # 解析每一行的 JSON 字符串
                    episodes_in_json[json_data['episode_id']] = json_data  # 将解析出来的字典添加到列表中
                    finish_status_in_json = json_data['finish_status']
                    fail_reason_in_json = json_data['fail_reason']
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")  # 错误处理
        # assert list(target_episodes.keys()) == list(episodes_in_json.keys())
        for key in target_episodes.keys():
            if key in episodes_in_json:
                origin = target_episodes[key]
                in_json = episodes_in_json[key]
                assert origin['instruction']['instruction_text'] == in_json['instruction_text']
                # assert origin['instruction']['instruction_tokens'] == in_json['instruction_tokens']
        assert value['finish_status'] == finish_status_in_json
        assert value['fail_reason'] == fail_reason_in_json

        rgb = value['episode_data']['camera_info']['pano_camera_0']['rgb']
        rgb_path = os.path.join(trajectory_path,"videos/chunk-000/observation.images.rgb/rgb.npy")
        rgb_n = np.load(rgb_path)
        assert np.array_equal(rgb, rgb_n)

        depth = value['episode_data']['camera_info']['pano_camera_0']['depth']
        depth_path = os.path.join(trajectory_path,"videos/chunk-000/observation.images.depth/depth.npy")
        depth_n = np.load(depth_path)
        assert np.array_equal(depth, depth_n)