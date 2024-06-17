from gym.envs.registration import register 

register(
	id='FrankaFlipBagel-v1',
	entry_point='franka_envs.envs:FrankaFlipEnv',
	max_episode_steps=25,
	) 