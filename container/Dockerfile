# G1 Brunson runner image (linux/arm64). Build from the HIM builder-kit root with this folder copied to
# brunson/ (the GitHub workflow does this):
#   docker buildx build --platform linux/arm64 --load -f brunson/Dockerfile -t g1-submission:local .
FROM ros:foxy-ros-base
SHELL ["/bin/bash", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake python3-colcon-common-extensions python3-pip \
    ros-foxy-rosidl-default-generators ros-foxy-rmw-fastrtps-cpp ros-foxy-std-srvs \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir "numpy==1.24.4" "onnxruntime==1.16.3"
WORKDIR /submission
COPY ros2/him_safety_interfaces /submission/interfaces
RUN source /opt/ros/foxy/setup.bash && colcon build --base-paths interfaces --packages-select him_safety_interfaces
COPY brunson/ /submission/brunson/
# stand-by balance network and its wrapper, from the builder kit (NVIDIA Open Model License, see walker/)
COPY examples/policies/walker/balance.onnx examples/policies/walker/LICENSE examples/policies/walker/provenance.json /submission/brunson/walker/
COPY robot_safety/balance_policy.py robot_safety/hardware_constants.py /submission/brunson/robot_safety/
RUN touch /submission/brunson/robot_safety/__init__.py \
    && cd /submission/brunson && python3 -c "import policy; policy.Policy('policy_weights.npz', freeze_wrists=True); from robot_safety.balance_policy import BalancePolicy; b = BalancePolicy('walker/balance.onnx'); b.warmup(); print('controllers load')"
ENTRYPOINT ["/bin/bash", "/submission/brunson/entrypoint.sh"]
