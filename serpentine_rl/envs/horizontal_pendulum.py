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
    near the middle of the crossbar for as long as possible.

    Observation vector (float32):
        0. left_cart_height          : world z-position at the left attachment
        1. left_cart_velocity        : vertical velocity of the left cart (m/s)
        2. right_cart_height         : world z-position at the right attachment
        3. right_cart_velocity       : vertical velocity of the right cart (m/s)
        4. crossbar_height           : height of the crossbar midpoint (m)
        5. crossbar_vertical_vel     : vertical velocity of the crossbar midpoint (m/s)
        6. rail_height_difference    : left minus right cart heights (m)
        7. crossbar_axis_z           : z-component of the bar’s x-axis (tilt indicator)
        8. crossbar_pitch_rate       : angular velocity about the world y-axis (rad/s)
        9. ring_displacement         : ring offset along the bar (m)
        10. ring_velocity            : ring velocity along the bar (m/s)
        11. partner_action           : sin-wave command applied to the right motor

    Action vector (float32):
        -1..1 motor command for the left rail actuator (right rail follows a sine wave).
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
        partner_wave_amplitude: float = 0.3,
        partner_wave_frequency: float = 0.45,
        cooperation_penalty: float = 0.0015,
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
            partner_wave_amplitude=partner_wave_amplitude,
            partner_wave_frequency=partner_wave_frequency,
            cooperation_penalty=cooperation_penalty,
            **kwargs,
        )

        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
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
        self.partner_wave_amplitude = partner_wave_amplitude
        self.partner_wave_frequency = partner_wave_frequency
        self.cooperation_penalty = cooperation_penalty

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
        ring_joint = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "ring_slide"
        )

        self._ring_qpos_idx = self.model.jnt_qposadr[ring_joint]
        self._ring_qvel_idx = self.model.jnt_dofadr[ring_joint]
        self._left_qpos_idx = self.model.jnt_qposadr[left_slider_joint]
        self._left_qvel_idx = self.model.jnt_dofadr[left_slider_joint]
        self._right_qpos_idx = self.model.jnt_qposadr[right_slider_joint]
        self._right_qvel_idx = self.model.jnt_dofadr[right_slider_joint]

        self._ring_limit = float(self.model.jnt_range[ring_joint, 1])
        self._nominal_crossbar_height = float(self.data.xpos[self._crossbar_body_id, 2])
        self._partner_phase = 0.0
        self._last_partner_action = 0.0
        self._last_left_action = 0.0

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
            self._ring_qpos_idx,
        ):
            qpos[idx] += self.np_random.uniform(low=-0.01, high=0.01)

        for idx in (
            self._left_qvel_idx,
            self._right_qvel_idx,
            self._ring_qvel_idx,
        ):
            qvel[idx] += self.np_random.uniform(low=-0.02, high=0.02)

        self.set_state(qpos, qvel)
        self._partner_phase = 0.0
        self._last_partner_action = self.partner_wave_amplitude * np.sin(self._partner_phase)
        self._last_left_action = 0.0
        return self._get_obs()

    # The base MujocoEnv signature uses Tuple[str, ...] but gymnasium expects Dict keys.
    def step(  # type: ignore[override]
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        left_action = float(np.clip(action[0], -1.0, 1.0))
        partner_action = float(
            np.clip(
                self.partner_wave_amplitude * np.sin(self._partner_phase), -1.0, 1.0
            )
        )
        ctrl = np.array([left_action, partner_action], dtype=np.float32)
        self._last_left_action = left_action
        self._last_partner_action = partner_action
        self.do_simulation(ctrl, self.frame_skip)
        self._partner_phase += (
            self.partner_wave_frequency * self.dt * float(self.frame_skip)
        )
        observation = self._get_obs()

        left_height = observation[0]
        left_vel = observation[1]
        right_height = observation[2]
        right_vel = observation[3]
        crossbar_height = observation[4]
        crossbar_vel = observation[5]
        rail_delta = observation[6]
        axis_z = observation[7]
        pitch_rate = observation[8]
        ring_disp = observation[9]
        ring_vel = observation[10]
        partner_action = observation[11]

        terminated = bool(
            not np.isfinite(observation).all()
            or np.abs(ring_disp) > self._ring_limit * 0.999
            or np.abs(axis_z) > self.max_tilt
            or crossbar_height < 0.08
            or crossbar_height > 1.4
        )

        ring_error = ring_disp - self.ring_target
        ring_penalty = self.ring_weight * ring_error**2
        tilt_penalty = self.tilt_weight * (rail_delta**2 + axis_z**2)
        height_error = crossbar_height - self._nominal_crossbar_height
        height_penalty = self.height_weight * height_error**2
        velocity_penalty = self.velocity_weight * (
            2.0 * ring_vel**2
            + pitch_rate**2
            + 0.3 * crossbar_vel**2
            + 0.5 * (left_vel**2 + right_vel**2)
        )
        control_penalty = 0.0005 * (left_action**2 + partner_action**2)
        cooperation_penalty = self.cooperation_penalty * float(
            (left_action - partner_action) ** 2
        )

        reward = 1.01 - (
            ring_penalty
            + tilt_penalty
            + height_penalty
            + velocity_penalty
            + control_penalty
            + cooperation_penalty
        )

        if self.render_mode == "human":
            self.render()

        info = {
            "ring_displacement": float(ring_disp),
            "ring_velocity": float(ring_vel),
            "crossbar_height": float(crossbar_height),
            "crossbar_axis_z": float(axis_z),
            "ring_height": float(self.data.site_xpos[self._ring_site_id, 2]),
            "left_action": float(left_action),
            "partner_action": float(partner_action),
        }

        # Truncation handled by external TimeLimit wrapper if used.
        return observation, float(reward), terminated, False, info

    def _get_obs(self) -> np.ndarray:
        left_height = self.data.site_xpos[self._left_site_id, 2]
        right_height = self.data.site_xpos[self._right_site_id, 2]

        left_vel = self.data.qvel[self._left_qvel_idx]
        right_vel = self.data.qvel[self._right_qvel_idx]

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
                right_height,
                right_vel,
                crossbar_height,
                crossbar_vel,
                left_height - right_height,
                axis_z,
                pitch_rate,
                ring_disp,
                ring_vel,
                self._last_partner_action,
            ],
            dtype=np.float32,
        )
        return observation


def make_env(**kwargs: Any) -> Env:
    """Factory helper used by scripts to instantiate the environment."""
    return HorizontalCooperativePendulumEnv(**kwargs)
