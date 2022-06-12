import gym 
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.env_util import make_vec_env
import os
import shared
import argparse
from car_racing_ppo import model_check

CNN_POLICY = 'CnnPolicy'

if __name__ == '__main__':
    input_args = argparse.ArgumentParser()
    input_args.add_argument('-t', '--time')
    input_args.add_argument('-m', '--mlp')
    input_args.add_argument('-c', '--cnn')
    args = input_args.parse_args()
    if args.time:
        TIMESTEPS = int(args.time)
    else:
        TIMESTEPS = 0
    env = gym.make(shared.ENV_NAME)
    env = DummyVecEnv([lambda: env])
    model = PPO(CNN_POLICY, env, verbose = 1, tensorboard_log=f'{shared.PPO_SAVE_PATH}/Logs/TensorLogs')
    model.learn(total_timesteps=TIMESTEPS)

    model.save(f'{shared.PPO_SAVE_PATH}/Logs/ppo_{CNN_POLICY}_{TIMESTEPS}')

    eval_output = evaluate_policy(model, env, n_eval_episodes=10, render=True)
    print(f"[evaluate_policy_model_{TIMESTEPS}]: Timesteps: {TIMESTEPS}, Evaluation output: {eval_output}")

    model_check(model, env, name=f"{TIMESTEPS}")