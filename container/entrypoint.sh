#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/foxy/setup.bash
source /submission/install/setup.bash
exec python3 /submission/brunson/brunson_node.py
