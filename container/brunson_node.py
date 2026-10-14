"""G1 Brunson runner controller (ROS 2 Foxy, HIM safety interface).

Arms through /safety/checker/arm, reads /safety/policy_observation (joint state) and /safety/policy_state
(JSON with the IMU quaternion wxyz and gyro), and publishes all 29 JointCommand slots to
/safety/precheck with the checker-issued session id, the received observation id and an increasing
sequence; stops publishing on /safety/abort.

The dribble controller needs the basketball's position and velocity (SENSOR_CONTRACT.md). The standard
runner provides joints and IMU only, so until a qualified ball input is connected this node runs its
stand-by behaviour:
  * first command: the measured pose (handover);
  * upright: the GR00T whole-body-control balance network shipped with the HIM builder kit
    (robot_safety/balance_policy.py, NVIDIA Open Model License) for legs and waist, arms held at their
    measured pose;
  * not upright (seated or lying starts): damping only, no position gain.
The trained dribble controller (policy.py, policy_weights.npz, assets/) and the head-camera tracker
(vision.py) are installed in the image for the camera integration.
"""
import json
import os
import sys

import numpy as np
import rclpy
from him_safety_interfaces.msg import Abort, JointCommand, RobotFeedback
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from robot_safety.balance_policy import LEG_KD, LEG_KP, BalancePolicy  # noqa: E402

FEEDBACK = "/safety/policy_observation"
STATE = "/safety/policy_state"
PRECHECK = "/safety/precheck"
ABORT = "/safety/abort"
ARM = "/safety/checker/arm"
CONTROL_DT = 0.02
DAMPING_KD = 2.0
N_LOWER = 15


def upright(quat_wxyz):
    w, x, y, z = quat_wxyz
    return 1 - 2 * (x * x + y * y) > 0.9  # body z axis within ~25 deg of vertical


class Brunson(Node):
    def __init__(self):
        super().__init__("g1_brunson")
        prof = json.load(open(os.path.join(HERE, "assets", "arena_g1_profile.json")))["joints"]
        self.kp = np.array([float(j["kp"]) for j in prof])
        self.kd = np.array([float(j["kd"]) for j in prof])
        self.balance = BalancePolicy(os.path.join(HERE, "walker", "balance.onnx"))
        self.balance.warmup()
        self.session_id = None
        self.sequence = 0
        self.done = False
        self.hold_q = None
        self.state = None
        self.mode = None
        self.last_control = None
        self.command = None
        feedback_qos = QoSProfile(depth=64, reliability=ReliabilityPolicy.BEST_EFFORT)
        command_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        abort_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                               durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(JointCommand, PRECHECK, command_qos)
        self.create_subscription(RobotFeedback, FEEDBACK, self.on_feedback, feedback_qos)
        self.create_subscription(String, STATE, self.on_state, feedback_qos)
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

    def on_state(self, message):
        try:
            self.state = json.loads(message.data)
        except ValueError:
            pass

    def control(self, q, dq):
        """50 Hz update; returns (q_target, kp, kd)."""
        quat = (self.state or {}).get("quat")
        gyro = (self.state or {}).get("gyro")
        if quat is None or gyro is None or not upright(quat):
            if self.mode != "damp":
                print("mode damp", flush=True)
            self.mode = "damp"
            self.balance.reset()
            return q.copy(), np.zeros(29), np.full(29, DAMPING_KD)
        if self.mode != "balance":
            print("mode balance", flush=True)
            self.mode = "balance"
            self.balance.reset()
            self.hold_q = q.copy()
        target = self.hold_q.copy()
        target[:N_LOWER] = self.balance.update(q, dq, quat, gyro)
        kp = self.kp.copy()
        kd = self.kd.copy()
        kp[:N_LOWER] = LEG_KP
        kd[:N_LOWER] = LEG_KD
        return target, kp, kd

    def on_feedback(self, message):
        if self.session_id is None or self.done:
            return
        q = np.array(message.q, float)
        dq = np.array(message.dq, float)
        now = message.received_at_secs
        if self.command is None:
            # handover: the first command is the measured pose
            self.command = (q.copy(), self.kp.copy(), self.kd.copy())
            self.last_control = now
        elif now - self.last_control >= CONTROL_DT - 1e-4:
            self.last_control = now
            self.command = self.control(q, dq)
        target, kp, kd = self.command
        n = len(message.q)
        command = JointCommand()
        command.session_id = self.session_id
        command.sequence = self.sequence
        command.observation_id = message.observation_id
        command.q = [float(v) for v in target[:n]]
        command.dq = [0.0] * n
        command.kp = [float(v) for v in kp[:n]]
        command.kd = [float(v) for v in kd[:n]]
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
