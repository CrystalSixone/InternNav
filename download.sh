# huggingface-cli download billzhao1030/vlnverse_scene \
#   --repo-type dataset \
#   --local-dir /cpfs/user/wangliuyi/vlnverse_scene \
#   --resume-download

export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download InternRobotics/InternVLA-N1 \
  --repo-type model \
  --local-dir /cpfs/user/wangliuyi/code/internnav_vlnverse/checkpoints/InternVLA-N1 \
  --resume-download