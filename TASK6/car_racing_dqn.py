import argparse
import gym
from collections import deque
from car_racing_dqn_context import CarRacingDQNContext
from dqn_train import preprocess_image
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Play CarRacing by the trained model.')
    parser.add_argument('-m', '--model', required=True, help='The `.h5` file of the trained model.')
    parser.add_argument('-e', '--episodes', type=int, default=1, help='The number of episodes should the model plays.')
    args = parser.parse_args()
    train_model = args.model
    play_episodes = args.episodes

    env = gym.make('CarRacing-v0')
    agent = CarRacingDQNContext(epsilon=0) # Set epsilon to 0 to ensure all actions are instructed by the agent
    agent.load(train_model)
    print('Episode, Scores(Time Frames), Total Rewards')
    results_table = []
    for e in range(play_episodes):
        init_state = env.reset()
        init_state = preprocess_image(init_state)

        total_reward = 0
        punishment_counter = 0
        state_frame_stack_queue = deque([init_state]*agent.frame_stack_size, maxlen=agent.frame_stack_size)
        time_frame_counter = 1
        
        while True:
            env.render()

            current_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
            action = agent.get_action(current_state_frame_stack)
            next_state, reward, done, info = env.step(action)

            total_reward += reward
            
            next_state = preprocess_image(next_state)
            state_frame_stack_queue.append(next_state)

            if done:
                print(f'{e+1}, {time_frame_counter}, {float(total_reward)}')
                results_table.append({'episode':e+1, 'frames':time_frame_counter, 'reward': float(total_reward)})
                break
            time_frame_counter += 1
    out = get_avg(results_table, 'reward')
    median = np.median([x['reward'] for x in results_table], axis=0)
    draw(to_df('episode', 'reward', results_table), 'Reward', 'Episode', f'Model: {get_file(args.model)}, Avg: {round(out, 3)}, Median: {round(median, 3)}')
