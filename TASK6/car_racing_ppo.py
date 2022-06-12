import argparse
import gym
from collections import deque
from car_racing_dqn_context import CarRacingDQNContext
from dqn_train import preprocess_image
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def generate_state_frame_stack_from_queue(deque):
    frame_stack = np.array(deque)
    # Move stack dimension to the channel dimension (stack, x, y) -> (x, y, stack)
    return np.transpose(frame_stack, (1, 2, 0))

def to_df(x_axi, y_axi, array):
    df_process = pd.DataFrame({
        'y_axis': [i[y_axi] for i in array],
        'x_axis': [i[x_axi] for i in array]
    })
    return df_process

def get_avg(array, element):
    ele = 0
    for i in array:
        ele += i[element]
    return ele/len(array)

def get_file(path):
    output = ''
    dot = False
    for i in range(len(path)-1, 0, -1):
        if dot == False and path[i] == '.':
            dot = True
        elif path[i] == '\\' or path[i] == '/':
            return  output
        elif dot == True:
            output = path[i] + output
    return output

def draw(df_process, x, y, title):
    fig, ax = plt.subplots()
    ax.set_title(title)
    sns.regplot(ax=ax, x=df_process["x_axis"], y=df_process["y_axis"], 
                line_kws={"color":"blue","alpha":0.7,"lw":2}, scatter_kws={"color": "blue"})
    ax.set_ylabel(x, size = 12)
    ax.set_xlabel(y, size = 12)
    plt.show()


def model_check(model, env, name='', episodes=10):
    obs = env.reset()
    results_table = []
    for episode in range(1, episodes+1):
        state = env.reset()
        done = False
        score = 0 
        
        while not done:
            env.render()
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            score+=reward
        results_table.append({'episode':episode, 'reward': float(score)})
    out = get_avg(results_table, 'reward')
    median = np.median([x['reward'] for x in results_table], axis=0)
    draw(to_df('episode', 'reward', results_table), 'Reward', 'Episode', f'Model: {get_file(name)}, Avg: {round(out, 3)}, Median: {round(median, 3)}')