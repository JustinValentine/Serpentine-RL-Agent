"""Cooperative dual-rail MuJoCo environment with a sliding ring objective."""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import mujoco
import numpy as np
from gymnasium import Env, spaces
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.utils import EzPickle


_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
_MODEL_PATH = os.path.join(_ASSETS_DIR, "horizontal_coop_pendulum.xml")

DEFAULT_CAMERA_CONFIG: Dict[str, Any] = {
    "distance": 3.0,
    "lookat": np.array([0.0, 0.0, 0.6]),
    "elevation": -20.0,
    "azimuth": 90.0,
}


class HorizontalCooperativePendulumEnv(MujocoEnv, EzPickle):
    """Two actuated vertical rails cradle a passive crossbar with a sliding ring payload.

    Each agent commands the vertical position of a cart that runs along its rail. The
    carts are linked by a rigid grey crossbar. A passive ring is mounted on that bar and
    can glide left/right as the bar tilts. The cooperative objective is to keep the ring
    near the middle of the crossbar for as long as possible. The partner (right) cart
    reacts to the learned cart by sampling random velocity magnitudes and pushing away
    from the left cart via a lightweight velocity servo, while a passive lateral slide
    adds slight horizontal compliance to the rail connection.

    Observation vector (float32):
        0. left_cart_height          : world z-position at the left attachment
        1. left_cart_velocity        : vertical velocity of the left cart (m/s)
        2. crossbar_height           : height of the crossbar midpoint (m)
        3. crossbar_vertical_vel     : vertical velocity of the crossbar midpoint (m/s)
        4. crossbar_axis_z           : z-component of the bar’s x-axis (tilt indicator)
        5. crossbar_pitch_rate       : angular velocity about the world y-axis (rad/s)
        6. ring_displacement         : ring offset along the bar (m)
        7. ring_velocity             : ring velocity along the bar (m/s)

    Action vector (float32):
        -1..1 motor command for the left rail actuator (right rail follows a reactive velocity servo).
    """

    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 25}

    def __init__(
        self,
        frame_skip: int = 2,
        max_tilt: float = 0.85,
        ring_target: float = 0.0,
        ring_weight: float = 6.0,
        tilt_weight: float = 1.0,
        height_weight: float = 0.6,
        velocity_weight: float = 0.03,
        partner_random_speed_min: float = 0.5,
        partner_random_speed_max: float = 2.2,
        partner_speed_interval_min: float = 0.25,
        partner_speed_interval_max: float = 0.75,
        partner_velocity_gain: float = 0.9,
        partner_torque_bias: float = 0.0,
        partner_min_torque: float = -1.0,
        partner_max_torque: float = 1.0,
        **kwargs: Any,
    ) -> None:
        EzPickle.__init__(
            self,
            frame_skip=frame_skip,
            max_tilt=max_tilt,
            ring_target=ring_target,
            ring_weight=ring_weight,
            tilt_weight=tilt_weight,
            height_weight=height_weight,
            velocity_weight=velocity_weight,
            partner_random_speed_min=partner_random_speed_min,
            partner_random_speed_max=partner_random_speed_max,
            partner_speed_interval_min=partner_speed_interval_min,
            partner_speed_interval_max=partner_speed_interval_max,
            partner_velocity_gain=partner_velocity_gain,
            partner_torque_bias=partner_torque_bias,
            partner_min_torque=partner_min_torque,
            partner_max_torque=partner_max_torque,
            **kwargs,
        )

        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        MujocoEnv.__init__(
            self,
            _MODEL_PATH,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        self.action_space = action_space
        self.max_tilt = max_tilt
        self.ring_target = ring_target
        self.ring_weight = ring_weight
        self.tilt_weight = tilt_weight
        self.height_weight = height_weight
        self.velocity_weight = velocity_weight
        if partner_random_speed_min < 0.0:
            raise ValueError("partner_random_speed_min must be non-negative.")
        if partner_random_speed_min > partner_random_speed_max:
            raise ValueError(
                "partner_random_speed_min must be <= partner_random_speed_max."
            )
        if partner_speed_interval_min <= 0.0:
            raise ValueError("partner_speed_interval_min must be > 0.")
        if partner_speed_interval_min > partner_speed_interval_max:
            raise ValueError(
                "partner_speed_interval_min must be <= partner_speed_interval_max."
            )
        self.partner_random_speed_min = partner_random_speed_min
        self.partner_random_speed_max = partner_random_speed_max
        self.partner_speed_interval_min = partner_speed_interval_min
        self.partner_speed_interval_max = partner_speed_interval_max
        self.partner_velocity_gain = partner_velocity_gain
        self.partner_torque_bias = partner_torque_bias
        self.partner_min_torque = float(partner_min_torque)
        self.partner_max_torque = float(partner_max_torque)
        if self.partner_min_torque > self.partner_max_torque:
            raise ValueError("partner_min_torque must be <= partner_max_torque.")

        self._left_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "left_cart"
        )
        self._right_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "right_cart"
        )
        self._crossbar_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "crossbar"
        )

        self._left_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "left_attach"
        )
        self._right_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "right_attach"
        )
        self._ring_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "ring_site"
        )

        left_slider_joint = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_slider"
        )
        right_slider_joint = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_slider"
        )
        right_slide_x_joint = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_slide_x"
        )
        ring_joint = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "ring_slide"
        )

        self._ring_qpos_idx = self.model.jnt_qposadr[ring_joint]
        self._ring_qvel_idx = self.model.jnt_dofadr[ring_joint]
        self._left_qpos_idx = self.model.jnt_qposadr[left_slider_joint]
        self._left_qvel_idx = self.model.jnt_dofadr[left_slider_joint]
        self._right_qpos_idx = self.model.jnt_qposadr[right_slider_joint]
        self._right_qvel_idx = self.model.jnt_dofadr[right_slider_joint]
        self._right_x_qpos_idx = self.model.jnt_qposadr[right_slide_x_joint]
        self._right_x_qvel_idx = self.model.jnt_dofadr[right_slide_x_joint]

        self._ring_limit = float(self.model.jnt_range[ring_joint, 1])
        self._right_vertical_limit = float(self.model.jnt_range[right_slider_joint, 1])
        self._nominal_crossbar_height = float(self.data.xpos[self._crossbar_body_id, 2])
        self._last_partner_target_velocity = 0.0
        self._last_partner_action = 0.0
        self._last_left_action = 0.0
        self._partner_speed_timer = 0.0
        self._partner_speed_magnitude = self.partner_random_speed_min

        # Gymnasium 0.29 expects MuJoCo <=2.x where `solver_iter` existed.
        # MuJoCo 3.x renamed it to `solver_niter`, so we expose a shim if needed.
        data_cls = type(self.data)
        if not hasattr(data_cls, "solver_iter") and hasattr(self.data, "solver_niter"):

            def _solver_iter(data: mujoco.MjData) -> int:
                niter = np.asarray(data.solver_niter)
                return int(niter[0]) if niter.size else 0

            setattr(data_cls, "solver_iter", property(_solver_iter))

    def reset_model(self) -> np.ndarray:
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        # Apply small perturbations to controllable joints only.
        for idx in (
            self._left_qpos_idx,
            self._right_qpos_idx,
            self._right_x_qpos_idx,
            self._ring_qpos_idx,
        ):
            qpos[idx] += self.np_random.uniform(low=-0.01, high=0.01)

        for idx in (
            self._left_qvel_idx,
            self._right_qvel_idx,
            self._right_x_qvel_idx,
            self._ring_qvel_idx,
        ):
            qvel[idx] += self.np_random.uniform(low=-0.02, high=0.02)

        self.set_state(qpos, qvel)
        self._partner_speed_timer = 0.0
        self._partner_speed_magnitude = self.partner_random_speed_min
        self._last_partner_target_velocity = 0.0
        self._last_partner_action = self._compute_partner_action()
        self._last_left_action = 0.0
        return self._get_obs()

    # The base MujocoEnv signature uses Tuple[str, ...] but gymnasium expects Dict keys.
    def step(  # type: ignore[override]
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        left_action = float(np.clip(action[0], -1.0, 1.0))
        partner_action = self._compute_partner_action()
        ctrl = np.array([left_action, partner_action], dtype=np.float32)
        self._last_left_action = left_action
        self._last_partner_action = partner_action
        self.do_simulation(ctrl, self.frame_skip)
        observation = self._get_obs()

        left_height = float(self.data.site_xpos[self._left_site_id, 2])
        left_vel = float(self.data.qvel[self._left_qvel_idx])
        crossbar_height = float(self.data.xpos[self._crossbar_body_id, 2])
        crossbar_vel = float(self.data.cvel[self._crossbar_body_id, 5])
        xmat = self.data.xmat[self._crossbar_body_id].reshape(3, 3)
        axis_z = float(xmat[2, 0])
        pitch_rate = float(self.data.cvel[self._crossbar_body_id, 1])
        ring_disp = float(self.data.qpos[self._ring_qpos_idx])
        ring_vel = float(self.data.qvel[self._ring_qvel_idx])
        partner_action = self._last_partner_action
        partner_target_velocity = self._last_partner_target_velocity

        terminated = bool(
            not np.isfinite(observation).all()
            or np.abs(ring_disp) > self._ring_limit * 0.999
            or np.abs(axis_z) > self.max_tilt
            or crossbar_height < 0.08
            or crossbar_height > 1.4
        )

        ring_error = ring_disp - self.ring_target
        ring_penalty = self.ring_weight * ring_error**2
        tilt_penalty = self.tilt_weight * (axis_z**2)
        height_error = crossbar_height - self._nominal_crossbar_height
        height_penalty = self.height_weight * height_error**2
        velocity_penalty = self.velocity_weight * (
            2.0 * ring_vel**2
            + pitch_rate**2
            + 0.3 * crossbar_vel**2
            + 0.5 * left_vel**2
        )
        control_penalty = 0.0005 * (left_action**2)

        reward = 1.01 - (
            ring_penalty
            + tilt_penalty
            + height_penalty
            + velocity_penalty
            + control_penalty
        )

        if self.render_mode == "human":
            self.render()

        info = {
            "ring_displacement": float(ring_disp),
            "ring_velocity": float(ring_vel),
            "crossbar_height": float(crossbar_height),
            "crossbar_axis_z": float(axis_z),
            "ring_height": float(self.data.site_xpos[self._ring_site_id, 2]),
            "left_height": float(left_height),
            "left_action": float(left_action),
            "partner_action": float(partner_action),
            "partner_target_velocity": float(partner_target_velocity),
            "partner_speed_magnitude": float(self._partner_speed_magnitude),
        }

        # Truncation handled by external TimeLimit wrapper if used.
        return observation, float(reward), terminated, False, info

    def _get_obs(self) -> np.ndarray:
        left_height = self.data.site_xpos[self._left_site_id, 2]
        left_vel = self.data.qvel[self._left_qvel_idx]

        crossbar_height = self.data.xpos[self._crossbar_body_id, 2]
        crossbar_vel = self.data.cvel[self._crossbar_body_id, 5]

        xmat = self.data.xmat[self._crossbar_body_id].reshape(3, 3)
        axis_z = xmat[2, 0]
        pitch_rate = self.data.cvel[self._crossbar_body_id, 1]

        ring_disp = self.data.qpos[self._ring_qpos_idx]
        ring_vel = self.data.qvel[self._ring_qvel_idx]

        observation = np.array(
            [
                left_height,
                left_vel,
                crossbar_height,
                crossbar_vel,
                axis_z,
                pitch_rate,
                ring_disp,
                ring_vel,
            ],
            dtype=np.float32,
        )
        return observation

    def _compute_partner_action(self) -> float:
        dt = self.dt * float(self.frame_skip)
        self._partner_speed_timer -= dt
        if self._partner_speed_timer <= 0.0:
            self._partner_speed_timer = float(
                self.np_random.uniform(
                    self.partner_speed_interval_min, self.partner_speed_interval_max
                )
            )
            self._partner_speed_magnitude = float(
                self.np_random.uniform(
                    self.partner_random_speed_min, self.partner_random_speed_max
                )
            )

        left_height = float(self.data.site_xpos[self._left_site_id, 2])
        right_height = float(self.data.site_xpos[self._right_site_id, 2])
        direction = np.sign(right_height - left_height)
        if abs(direction) < 1e-3:
            direction = float(self.np_random.choice([-1.0, 1.0]))

        right_pos = float(self.data.qpos[self._right_qpos_idx])
        margin = 0.04
        if right_pos > self._right_vertical_limit - margin:
            direction = -1.0
        elif right_pos < -self._right_vertical_limit + margin:
            direction = 1.0

        height_gap = abs(left_height - right_height)
        gap_scale = np.clip(
            1.0 + 0.8 * (height_gap / max(self._right_vertical_limit, 1e-6)),
            1.0,
            2.0,
        )
        target_velocity = direction * self._partner_speed_magnitude * gap_scale
        self._last_partner_target_velocity = float(target_velocity)

        current_velocity = float(self.data.qvel[self._right_qvel_idx])
        torque_cmd = (
            self.partner_torque_bias
            + self.partner_velocity_gain * (target_velocity - current_velocity)
        )
        bounded = np.clip(torque_cmd, self.partner_min_torque, self.partner_max_torque)
        return float(bounded)


def make_env(**kwargs: Any) -> Env:
    """Factory helper used by scripts to instantiate the environment."""
    return HorizontalCooperativePendulumEnv(**kwargs)
