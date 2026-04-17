"""
Joint Signal AV — MARL signalling experiment
RQ1: Do jointly-trained AVs develop behavioural signalling?
Metric: separation score rho = normalised MI(style, type)
Usage: uv run train.py
"""

import time

import gymnasium as gym
import highway_env  # noqa: F401  registers envs
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

N_AGENTS   = 4
N_TYPES    = 3    # 0=cautious  1=normal  2=assertive
N_STYLES   = 3    # 0=smooth    1=neutral  2=sharp
N_ACTIONS  = 5    # DiscreteMetaAction: LANE_LEFT IDLE LANE_RIGHT FASTER SLOWER

OBS_VEHS   = 5    # ego + 4 neighbours (highway-env default)
OBS_FEATS  = 5    # presence x y vx vy
RAW_OBS    = OBS_VEHS * OBS_FEATS   # 25
AUG_OBS    = RAW_OBS + N_STYLES     # 28  (raw + own-style one-hot)
HIDDEN     = 128

TIME_BUDGET   = 600   # wall-clock seconds
ROLLOUT_LEN   = 64    # env steps per agent per rollout (256 total transitions)
PPO_EPOCHS    = 4
MINIBATCH     = 64
GAMMA         = 0.99
GAE_LAMBDA    = 0.95
CLIP_EPS      = 0.2
ENT_COEF      = 0.02
VF_COEF       = 0.5
LR            = 3e-4

# Style execution effect: smooth slows slightly, sharp speeds slightly.
# Which style maps to which driving type is NOT hard-coded — it emerges from training.
STYLE_SPEED_FACTOR = np.array([0.85, 1.00, 1.15], dtype=np.float32)

# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def make_env():
    env = gym.make("highway-v0", render_mode=None)
    env.unwrapped.configure({
        "controlled_vehicles": N_AGENTS,
        "vehicles_count": 20,
        "lanes_count":    4,
        "duration":       40,
        "observation": {
            "type": "MultiAgentObservation",
            "observation_config": {
                "type":           "Kinematics",
                "vehicles_count": OBS_VEHS,
                "features":       ["presence", "x", "y", "vx", "vy"],
                "normalize":      True,
                "absolute":       False,
                "order":          "sorted",
            },
        },
        "action": {
            "type": "MultiAgentAction",
            "action_config": {"type": "DiscreteMetaAction"},
        },
        "reward_speed_range": [20, 30],
        "collision_reward":  -5.0,
        "high_speed_reward":  0.4,
    })
    return env


def reset_with_types(env):
    obs, _ = env.reset()
    # MultiAgentObservation returns a tuple of N arrays each (OBS_VEHS, OBS_FEATS)
    obs = np.stack([np.array(o, dtype=np.float32) for o in obs])  # (N_AGENTS, OBS_VEHS, OBS_FEATS)
    types = np.random.randint(0, N_TYPES, size=N_AGENTS)
    return obs, types


def apply_styles(env, styles):
    """Modify controlled vehicles' target speeds according to chosen style."""
    for i, v in enumerate(env.unwrapped.controlled_vehicles):
        v.target_speed = float(np.clip(v.target_speed * STYLE_SPEED_FACTOR[styles[i]], 15.0, 40.0))


def augment_obs(raw_obs, styles):
    """Append own-style one-hot to each agent's flat observation. (N_AGENTS, AUG_OBS)"""
    flat = raw_obs.reshape(N_AGENTS, -1)
    style_onehots = np.eye(N_STYLES, dtype=np.float32)[styles]
    return np.concatenate([flat, style_onehots], axis=1)


def get_agent_rewards(env):
    """Per-vehicle reward from individual vehicle states."""
    rewards = np.zeros(N_AGENTS, dtype=np.float32)
    for i, v in enumerate(env.unwrapped.controlled_vehicles):
        speed_r = np.clip((v.speed - 20.0) / 10.0, 0.0, 1.0) * 0.4
        crash_r = -5.0 if v.crashed else 0.0
        rewards[i] = speed_r + crash_r
    return rewards


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class AgentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(AUG_OBS, HIDDEN)
        self.l2 = nn.Linear(HIDDEN, HIDDEN)
        self.action_head = nn.Linear(HIDDEN, N_ACTIONS)
        self.style_head  = nn.Linear(HIDDEN, N_STYLES)
        self.value_head  = nn.Linear(HIDDEN, 1)

    def __call__(self, x):
        h = mx.maximum(self.l1(x), 0)
        h = mx.maximum(self.l2(h), 0)
        return self.action_head(h), self.style_head(h), self.value_head(h)[..., 0]


# ---------------------------------------------------------------------------
# PPO helpers
# ---------------------------------------------------------------------------

def log_prob_discrete(logits, actions):
    log_p = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return log_p[mx.arange(logits.shape[0]), actions]


def entropy_discrete(logits):
    log_p = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return -mx.sum(mx.exp(log_p) * log_p, axis=-1)


def compute_gae(rewards, values, dones):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        nv = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + GAMMA * nv * (1 - dones[t]) - values[t]
        gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * gae
        adv[t] = gae
    return adv, adv + np.array(values[:T])


def ppo_update(model, optimizer, buf):
    obs_a    = np.array(buf["obs"],     dtype=np.float32)
    act_a    = np.array(buf["actions"], dtype=np.int32)
    sty_a    = np.array(buf["styles"],  dtype=np.int32)
    olp_act  = np.array(buf["lp_act"],  dtype=np.float32)
    olp_sty  = np.array(buf["lp_sty"],  dtype=np.float32)
    adv_a, ret_a = compute_gae(buf["rewards"], buf["values"], buf["dones"])
    adv_a = (adv_a - adv_a.mean()) / (adv_a.std() + 1e-8)

    T = len(obs_a)
    for _ in range(PPO_EPOCHS):
        idx = np.random.permutation(T)
        for start in range(0, T, MINIBATCH):
            mb = idx[start : start + MINIBATCH]
            if len(mb) < 4:
                continue
            obs_mb  = mx.array(obs_a[mb])
            act_mb  = mx.array(act_a[mb])
            sty_mb  = mx.array(sty_a[mb])
            olpa_mb = mx.array(olp_act[mb])
            olps_mb = mx.array(olp_sty[mb])
            adv_mb  = mx.array(adv_a[mb])
            ret_mb  = mx.array(ret_a[mb])

            def loss_fn(model):
                al, sl, v = model(obs_mb)
                # action clip loss
                nlpa = log_prob_discrete(al, act_mb)
                r_a  = mx.exp(nlpa - olpa_mb)
                L_act = -mx.mean(mx.minimum(r_a * adv_mb, mx.clip(r_a, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_mb))
                # style clip loss
                nlps = log_prob_discrete(sl, sty_mb)
                r_s  = mx.exp(nlps - olps_mb)
                L_sty = -mx.mean(mx.minimum(r_s * adv_mb, mx.clip(r_s, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_mb))
                # value + entropy
                L_val = mx.mean((v - ret_mb) ** 2)
                L_ent = mx.mean(entropy_discrete(al)) + mx.mean(entropy_discrete(sl))
                return L_act + L_sty + VF_COEF * L_val - ENT_COEF * L_ent

            loss, grads = nn.value_and_grad(model, loss_fn)(model)
            optimizer.update(model, grads)
            mx.eval(model.parameters())


# ---------------------------------------------------------------------------
# Separation score rho
# ---------------------------------------------------------------------------

def compute_rho(pairs):
    """
    rho = MI(style, type) / H(type)
    0 = no signalling, 1 = perfect signalling
    """
    if not pairs:
        return 0.0
    styles = np.array([p[0] for p in pairs])
    types  = np.array([p[1] for p in pairs])
    joint  = np.zeros((N_STYLES, N_TYPES), dtype=np.float64)
    for s, t in zip(styles, types):
        joint[s, t] += 1
    joint /= joint.sum()
    p_s = joint.sum(axis=1)
    p_t = joint.sum(axis=0)

    def h(p):
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))

    mi = h(p_s) + h(p_t) - h(joint.flatten())
    ht = h(p_t)
    return mi / ht if ht > 1e-10 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

t_start = time.time()
np.random.seed(42)
mx.random.seed(42)

env   = make_env()
model = AgentNet()
mx.eval(model.parameters())
optimizer = optim.Adam(learning_rate=LR)

raw_obs, types = reset_with_types(env)
styles = np.zeros(N_AGENTS, dtype=np.int32)

style_type_log = []
ep_rewards     = []
ep_r           = np.zeros(N_AGENTS)
step           = 0
episode        = 0
total_time     = 0.0

buf = {"obs": [], "actions": [], "styles": [], "lp_act": [], "lp_sty": [],
       "rewards": [], "values": [], "dones": []}

print(f"N_AGENTS={N_AGENTS}  N_TYPES={N_TYPES}  N_STYLES={N_STYLES}  budget={TIME_BUDGET}s")

def softmax_np(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

while True:
    t0 = time.time()

    apply_styles(env, styles)
    aug = augment_obs(raw_obs, styles)

    obs_mx = mx.array(aug)
    al, sl, v = model(obs_mx)
    mx.eval(al, sl, v)

    al_np = np.array(al); sl_np = np.array(sl); v_np = np.array(v)
    ap = softmax_np(al_np); sp = softmax_np(sl_np)

    actions    = np.array([np.random.choice(N_ACTIONS, p=ap[i]) for i in range(N_AGENTS)])
    new_styles = np.array([np.random.choice(N_STYLES,  p=sp[i]) for i in range(N_AGENTS)])
    lp_act = np.log(np.array([ap[i, actions[i]]    for i in range(N_AGENTS)]) + 1e-10)
    lp_sty = np.log(np.array([sp[i, new_styles[i]] for i in range(N_AGENTS)]) + 1e-10)

    raw_obs_next, _, terminated, truncated, info = env.step(tuple(int(a) for a in actions))
    raw_obs_next = np.stack([np.array(o, dtype=np.float32) for o in raw_obs_next])
    rewards = get_agent_rewards(env)
    done    = bool(terminated or truncated)

    style_type_log.extend(zip(new_styles.tolist(), types.tolist()))
    ep_r += rewards

    for i in range(N_AGENTS):
        buf["obs"].append(aug[i])
        buf["actions"].append(int(actions[i]))
        buf["styles"].append(int(new_styles[i]))
        buf["lp_act"].append(float(lp_act[i]))
        buf["lp_sty"].append(float(lp_sty[i]))
        buf["rewards"].append(float(rewards[i]))
        buf["values"].append(float(v_np[i]))
        buf["dones"].append(float(done))

    styles  = new_styles
    raw_obs = raw_obs_next

    if done:
        ep_rewards.append(float(ep_r.mean()))
        ep_r = np.zeros(N_AGENTS)
        raw_obs, types = reset_with_types(env)
        styles = np.zeros(N_AGENTS, dtype=np.int32)
        episode += 1

    total_time += time.time() - t0
    step += 1

    if len(buf["obs"]) >= ROLLOUT_LEN * N_AGENTS:
        ppo_update(model, optimizer, buf)
        for k in buf:
            buf[k].clear()

        rho = compute_rho(style_type_log[-20000:])
        mean_r = np.mean(ep_rewards[-20:]) if ep_rewards else 0.0
        pct = 100 * min(total_time / TIME_BUDGET, 1.0)
        print(f"\rep {episode:04d} ({pct:.1f}%) rho={rho:.4f} mean_r={mean_r:.3f}    ", end="", flush=True)

    if total_time >= TIME_BUDGET:
        break

print()
env.close()

rho_final  = compute_rho(style_type_log)
mean_r_final = np.mean(ep_rewards[-50:]) if ep_rewards else 0.0

print("---")
print(f"rho:              {rho_final:.6f}")
print(f"mean_ep_reward:   {mean_r_final:.4f}")
print(f"episodes:         {episode}")
print(f"steps:            {step}")
print(f"training_seconds: {total_time:.1f}")
print(f"n_agents:         {N_AGENTS}")
print(f"n_types:          {N_TYPES}")
print(f"n_styles:         {N_STYLES}")
