#!/usr/bin/env bash
# One-shot setup of a fresh HIM "standard" GPU machine for this project (run on the machine).
set -euo pipefail
cd /workspace
(apt-get update -qq && apt-get install -y -qq libegl1 libgl1 libglib2.0-0 ffmpeg) > /tmp/apt.log 2>&1 || true
if [ ! -x /workspace/venv/bin/python ]; then
  uv venv -q --python 3.11 /workspace/venv
  source /workspace/venv/bin/activate
  uv pip install -q -r /workspace/g1dribble/requirements-train.txt pillow --index-strategy unsafe-best-match
fi
source /workspace/venv/bin/activate
# Unitree robot description (for build_model.py) and Unitree's MuJoCo RL model
mkdir -p /workspace/assets && cd /workspace/assets
if [ ! -d unitree_ros ]; then
  git clone -q --depth 1 --filter=blob:none --sparse https://github.com/unitreerobotics/unitree_ros.git
  (cd unitree_ros && git sparse-checkout set robots/g1_description -q)
fi
mkdir -p mjlab_g1 && cp /workspace/g1dribble/mjlab_g1.xml mjlab_g1/g1.xml
cat > /workspace/g1dribble/bg.sh <<'EOF'
#!/usr/bin/env bash
log=$1; shift
cd /workspace/g1dribble
setsid nohup bash -c "source /workspace/venv/bin/activate; export MUJOCO_GL=egl; $*" > "$log" 2>&1 < /dev/null &
echo "started: $* -> $log"
EOF
chmod +x /workspace/g1dribble/bg.sh /workspace/g1dribble/clip_watcher.sh
python -c "import jax, mujoco, brax, mujoco_playground; print('SETUP OK', jax.devices())"
