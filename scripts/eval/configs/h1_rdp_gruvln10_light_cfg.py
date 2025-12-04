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
        ckpt_path='checkpoints/20251202_rdp_gruvln10_train_epoch50/ckpts/checkpoint-2750',
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
        task_name='20251204_rdp_epoch50_gruvln10_original_eval_lightDebug',
        task_settings={
            'env_num': 2,
            'use_distributed': False,
            'proc_num': 2,
        },
        scene=SceneCfg(
            scene_type='grscene_original',
            scene_data_dir='data/scene_data/grutopia10_original',
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
            'split_data_types': ['val_seen', 'val_unseen'],
            'filter_stairs': False,
        },
    ),
    eval_type='vln_distributed',
    eval_settings={
        'save_to_json': True,
        'vis_output': True,
        'use_agent_server': False,
    },
)
