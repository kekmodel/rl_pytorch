## Deep RL Algorithms in PyTorch

### Models
- DQN
- Dueling Double DQN
- Categorical DQN (C51)
- Categotical Dueling Double DQN
- Proximal Policy Optimization (PPO)
	+ discrete (episodic, n-step)
- Soft Actor-Critic (SAC)
	+ debugging
<br>

### Exploration
- Random Network Distillation (RND)
<br>

### Experiments
The result of passing the environment-defined "solving" criteria.
- **Dueling Double DQN**
	+ Only one hyperparameter "UP_COEF" was adjusted.
###### CartPole-v0
<div align="center">
  <img src="./image/CartPole-v0.gif" width="50%"><img src="./image/CartPole-v0_reward_curve.png" width="50%">
</div>

###### CartPole-v1
<div align="center">
  <img src="./image/CartPole-v1.gif" width="50%"><img src="./image/CartPole-v1_reward_curve.png" width="50%">
</div>

###### MountainCar-v0
<div align="center">
  <img src="./image/MountainCar-v0.gif" width="50%"><img src="./image/MountainCar-v0_reward_curve.png" width="50%">
</div>

###### LunarLander-v2
<div align="center"> 
  <img src="./image/LunarLander-v2.gif" width="50%"><img src="./image/LunarLander-v2_reward_curve.png" width="50%">
</div>
<br>

### TODO
- Quantile Regression DQN (QR DQN)
- Implicit Quantile Networks (IQN)
- Intrinsic Curiosity Module (ICM)
- Rainbow
- Parametric DQN
- Proximal Policy Optimization (PPO)
	+ continuous
- Deep Deterministic Policy Gradient (DDPG)
- MCTS Net
- Parallel Models 
	+ Ape-X
	+ R2D2
	+ PAAC
