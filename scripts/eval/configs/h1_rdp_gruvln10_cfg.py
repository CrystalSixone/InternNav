from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import (
    EnvCfg,
    EvalCfg,
    EvalDatasetCfg,
    SceneCfg,
    TaskCfg,
)

eval_cfg = EvalCfg(
    agent=AgentCfg(
        server_port=8087,
        model_name='rdp',
        ckpt_path='checkpoints/r2r/fine_tuned/rdp/checkpoint-104150',
        model_settings={},
    ),
    env=EnvCfg(
        env_type='internutopia',
        env_settings={
            'use_fabric': False,
            'headless': True,
        },
    ),
    task=TaskCfg(
        task_name='20251202_rdp_gruvln10_eval_trainSet',
        task_settings={
            'env_num': 1,
            'use_distributed': False,
            'proc_num': 1,
        },
        scene=SceneCfg(
            scene_type='grscene',
            scene_data_dir='data/scene_data/grutopia10',
        ),
        robot_name='h1',
        robot_usd_path='data/Embodiments/vln-pe/h1/h1_vln_pointcloud.usd',
        camera_resolution=[256, 256],  # (W,H)
        camera_prim_path='torso_link/h1_pano_camera_0',
    ),
    dataset=EvalDatasetCfg(
        dataset_type="kujiale",
        dataset_settings={
            'base_data_dir': 'data/vln_pe/raw_data/gruvln10',
            'split_data_types': ['train'],
            'filter_stairs': False,
        },
    ),
)
