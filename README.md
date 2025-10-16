# Serpentine RL Agent – Dual-Rail Ring Balance

This repository hosts a custom MuJoCo environment and PPO training harness where two vertically actuated carts hold a passive grey crossbar. A free-moving ring rides on that bar; the learning agent controls the left cart while the right cart reacts by sampling random velocity magnitudes and moving away from the learner, with a passive lateral slide adding compliance. The objective is to keep the ring centred for as long as possible.

## Environment Overview

- **Engine:** MuJoCo (`gymnasium` bindings)  
- **Actuators:** one trainable motor on the left cart and a reactive velocity servo driving the right cart (random speed bursts that push away from the learner)  
- **Passive mechanics:** rigid grey crossbar connecting both carts and a passive ring constrained to slide along the bar, with a small lateral compliance on the right cart  
- **Observation (8 values):** left rail height & velocity, crossbar height & vertical velocity, bar tilt (`z`-axis component), bar pitch rate, and the ring displacement & velocity  
- **Action (1 value):** continuous motor command in `[-1, 1]` for the left rail motor (the right rail follows the reactive partner servo)  
- **Reward:** starts at `1.01` per step and subtracts penalties for ring displacement, bar tilt, crossbar height error, velocity energy, and motor effort. Episodes terminate when the ring nears the end stops, the bar tilts too far from horizontal, or the crossbar leaves its safe vertical band.

The MuJoCo model lives in `serpentine_rl/envs/assets/horizontal_coop_pendulum.xml`. It builds twin vertical rails linked by an equality-constrained crossbar and adds a sliding “ring” carriage that must be balanced near the centre while the partner rail oscillates.

## Getting Started

```bash
# (Optional) create a fresh virtual environment
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### Train the cooperative agent

```bash
PYTHONPATH=. python scripts/train_horizontal_pendulum.py \
    --total-timesteps 200000 \
    --log-dir runs/horizontal_pendulum
```

The script logs TensorBoard summaries and checkpoints to `runs/<timestamp>/`. Use `--render-eval` to watch evaluation rollouts during training. Episodes are capped at 1,000 steps by the `TimeLimit` wrapper.

## Sample Rollout

![Policy balancing the ring](renders/eval_gifs/episode_01.gif)

### Reward shaping

The step reward is `1.01` minus a bundle of quadratic penalties:

- ring displacement relative to the centre (`ring_weight * (ring_disp - target)^2`)
- bar tilt penalty (`tilt_weight * axis_z^2`)
- crossbar height error versus its nominal resting height
- kinetic energy proxies for the ring, crossbar pitch/vertical velocity, and the left rail velocity
- mild `L2` control effort on the learner-controlled actuator

The scripted partner actuator now samples higher random velocity magnitudes and uses a proportional servo to push the right cart away from the left cart. Tune `partner_random_speed_min/max`, `partner_speed_interval_min/max`, `partner_velocity_gain`, `partner_torque_bias`, and the torque bounds (`partner_min_torque`, `partner_max_torque`) to shape how reactive or forceful the partner feels. The cart’s passive lateral slide (`right_slide_x`) is unactuated and only adds compliance.

Episodes end early if the ring approaches the limits, the bar tips beyond `max_tilt`, or the crossbar moves outside `[0.08, 1.4]` m.

### Evaluate a trained policy

```bash
PYTHONPATH=. python scripts/evaluate_horizontal_pendulum.py \
    --model-path runs/horizontal_pendulum/horizontal_pendulum_ppo.zip \
    --episodes 3 --deterministic --sleep 0.02
```

### Evaluate and record GIFs

```bash
PYTHONPATH=. python scripts/evaluate_and_record_horizontal_pendulum.py \
    --model-path runs/horizontal_pendulum \
    --episodes 3 --deterministic --output-dir renders/eval_gifs
```

The recorder accepts a specific checkpoint path (e.g. `.../best_model.zip`) or a directory and will pick a suitable `.zip` file automatically, preferring `best_model.zip`, `latest.zip`, then `horizontal_pendulum_ppo.zip`.
