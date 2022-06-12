import cv2
from car_racing_dqn_context import CarRacingDQNContext
import numpy as np
from collections import deque
import argparse
import gym
import shared

START_NEGATIVE_COUNTER = 30
MAXIMUM_NEGATIVE_REWORD = 25
TRAIN_SIZE = 16
UPDATE_TARGET_MODEL_FREQUENCY = 5
SAVE_TRAINING_FREQUENCY = 10

def preprocess_image(img):
    #change colors to RGB->GRAY
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #change type
    img = img.astype(float)
    #lower the values to 0/1
    img /= 255.0
    return img

class Session:
    def __init__(self, state, stack_size) -> None:
        self.reward = 0
        self.neg_reward = 0
        self.frame_queue = deque([state]*stack_size, maxlen=stack_size)
        self.couter = 1
        self.status = False

if __name__ == '__main__':
    input_args = argparse.ArgumentParser()
    input_args.add_argument('-m', '--model')
    input_args.add_argument('-s', '--start')
    input_args.add_argument('-e', '--end')
    input_args.add_argument('-f', '--frames')
    input_args.add_argument('-r', '--render')
    args = input_args.parse_args()

    env = gym.make(shared.ENV_NAME)
    context = CarRacingDQNContext()
    
    #init episodes
    START = args.start if args.start else 1
    STOP = args.end if args.end else 500
    IGNORE_FRAME = args.frames if args.frames else 2

    for episode in range(START, STOP+1):
        entry_state = env.reset()
        entry_state = preprocess_image(entry_state)

        session = Session(entry_state, context.frame_stack_size)
        print(session.couter,session.reward)
        while True:
            env.render()
            print(f"Episode: {episode}")
            # swap the dimention of the stack to the channel dimention
            current_state_frames = np.transpose(np.array(session.frame_queue), (1,2,0))

            action = context.get_action(current_state_frames)
            
            #Forward the frames with the generated action
            episode_reward = 0
            for i in range(IGNORE_FRAME + 1):
                next_step, local_reward, session.status, info =\
                        env.step(action)
                episode_reward += local_reward
                if session.status:
                    break
            print(session.couter, session.reward)
            #Introduce an additional bunus if model uses full gas
            if action[1] == 1 and action[2] == 0:
                episode_reward *= 1.5

            #Update tee negative         
            session.neg_reward = session.neg_reward + 1\
                    if session.couter > START_NEGATIVE_COUNTER and episode_reward < 0 else 0

            #Increase the total session reward
            session.reward += episode_reward

            next_step = preprocess_image(next_step)
            session.frame_queue.append(next_step)
            next_frame_queue = np.transpose(np.array(session.frame_queue), (1,2,0))

            #Memorize the step
            context.memorize(\
                current_state_frames, action, episode_reward, next_frame_queue, session.status)

            if session.status or\
               session.neg_reward > MAXIMUM_NEGATIVE_REWORD or\
               session.reward < 0:
                print(f'Episode: {episode}/{STOP}, Scores(Time Frames): {session.reward}, Total Rewards(adjusted): {session.reward}, Epsilon: {context.epsilon}')
                break
            
            #If memory is greater then training size start replaying
            if len(context.memory) > TRAIN_SIZE:
                context.get_replay(TRAIN_SIZE)

            #Increase the session counter
            session.couter += 1
        if episode % UPDATE_TARGET_MODEL_FREQUENCY == 0:
            context.update_output()
        if episode % SAVE_TRAINING_FREQUENCY == 0:
            context.save(f'./save/trial_{episode}.h5')