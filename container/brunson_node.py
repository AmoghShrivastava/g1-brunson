"""G1 Brunson runner controller (ROS 2 Foxy, HIM safety interface).

Arms through /safety/checker/arm, reads /safety/policy_observation (joint state) and publishes all 29
JointCommand slots to /safety/precheck with the checker-issued session id, the received observation id
and an increasing sequence; stops publishing on /safety/abort.

The dribble controller needs the basketball's position and velocity (SENSOR_CONTRACT.md).  The standard
runner provides joints and IMU only, so until a qualified ball input is connected this node runs the
controller's stand-by mode: it holds the joint pose measured when the session started, with the
profile PD gains, from any starting posture.  The trained controller (policy.py, policy_weights.npz,
stance and profile under assets/) and the head-camera tracker (vision.py) are installed in the image
for that integration.
"""
import json
import os
import sys

import rclpy
from him_safety_interfaces.msg import Abort, JointCommand, RobotFeedback
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_srvs.srv import Trigger

HERE = os.path.dirname(os.path.abspath(__file__))
FEEDBACK = "/safety/policy_observation"
PRECHECK = "/safety/precheck"
ABORT = "/safety/abort"
ARM = "/safety/checker/arm"


class Brunson(Node):
    def __init__(self):
        super().__init__("g1_brunson")
        prof = json.load(open(os.path.join(HERE, "assets", "arena_g1_profile.json")))["joints"]
        self.kp = [float(j["kp"]) for j in prof]
        self.kd = [float(j["kd"]) for j in prof]
        self.session_id = None
        self.sequence = 0
        self.done = False
        self.hold_q = None
        feedback_qos = QoSProfile(depth=64, reliability=ReliabilityPolicy.BEST_EFFORT)
        command_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        abort_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                               durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(JointCommand, PRECHECK, command_qos)
        self.create_subscription(RobotFeedback, FEEDBACK, self.on_feedback, feedback_qos)
        self.create_subscription(Abort, ABORT, self.on_abort, abort_qos)
        self.arm = self.create_client(Trigger, ARM)

    def wait_for_session(self):
        if not self.arm.wait_for_service(timeout_sec=10.0):
            print("arm service did not appear", flush=True)
            return False
        future = self.arm.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        result = future.result()
        if result is None or not result.success or not result.message:
            print("arm failed", flush=True)
            return False
        self.session_id = result.message
        print("armed", flush=True)
        return True

    def on_abort(self, message):
        print(f"abort {message.reason}", flush=True)
        self.done = True

    def on_feedback(self, message):
        if self.session_id is None or self.done:
            return
        if self.hold_q is None:
            self.hold_q = [float(v) for v in message.q]
        n = len(message.q)
        command = JointCommand()
        command.session_id = self.session_id
        command.sequence = self.sequence
        command.observation_id = message.observation_id
        command.q = list(self.hold_q)
        command.dq = [0.0] * n
        command.kp = list(self.kp[:n])
        command.kd = list(self.kd[:n])
        command.tau = [0.0] * n
        self.pub.publish(command)
        self.sequence += 1


def main():
    rclpy.init()
    node = Brunson()
    try:
        if not node.wait_for_session():
            return 2
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
