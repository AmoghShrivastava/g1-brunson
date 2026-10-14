# G1 Brunson runner image (linux/arm64). Build from the HIM builder-kit root with this folder copied to
# brunson/ (the GitHub workflow does this):
#   docker buildx build --platform linux/arm64 --load -f brunson/Dockerfile -t g1-submission:local .
FROM ros:foxy-ros-base
SHELL ["/bin/bash", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake python3-colcon-common-extensions \
    ros-foxy-rosidl-default-generators ros-foxy-rmw-fastrtps-cpp ros-foxy-std-srvs \
    python3-numpy \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /submission
COPY ros2/him_safety_interfaces /submission/interfaces
RUN source /opt/ros/foxy/setup.bash && colcon build --base-paths interfaces --packages-select him_safety_interfaces
COPY brunson/ /submission/brunson/
RUN python3 -c "import sys; sys.path.insert(0, '/submission/brunson'); import policy; policy.Policy('/submission/brunson/policy_weights.npz', freeze_wrists=True); print('controller loads')"
ENTRYPOINT ["/bin/bash", "/submission/brunson/entrypoint.sh"]
