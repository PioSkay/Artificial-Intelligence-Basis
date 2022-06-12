from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from collections import deque
import numpy as np
import random

FRAME_STACK_SIZE = 3
NETWORK_LEARNING_RATE = 0.001

class Training:
    def __init__(self) -> None:
        self.state = []
        self.target = []

class CarRacingDQNContext:
    def __init__(self, epsilon=0.95) -> None:
        self.frame_stack_size = 3
        self.actions = [
            # (Steering Wheel, Gas, Break)
            # This is the avalible action space
            (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2),
            (-1, 1,   0), (0, 1,   0), (1, 1,   0),
            (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2),
            (-1, 0,   0), (0, 0,   0), (1, 0,   0)
        ]
        self.epsilon = epsilon
        self.epsilon_min = 0.1
        self.epsilon_decrease = 0.9999
        self.discount_rate = 0.95
        self.memory = deque(maxlen=5000)
        self.model = self.get_model()
        self.output_model = self.get_model()
        self.update_output()

    def update_output(self):
        self.output_model.set_weights(self.model.get_weights())

    def get_model(self):
        model = Sequential()
        model.add(Conv2D(filters=6, kernel_size=(7, 7), strides=3, activation='relu', input_shape=(96, 96, FRAME_STACK_SIZE)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=12, kernel_size=(4, 4), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(216, activation='relu'))
        model.add(Dense(len(self.actions), activation=None))
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=NETWORK_LEARNING_RATE, epsilon=1e-7))
        return model

    def get_action(self, state):
        #Generate random action value
        action_id = random.randrange(len(self.actions))
        #Get the action value is possible
        if np.random.rand() > self.epsilon:
            output_action_value = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
            action_id = np.argmax(output_action_value[0])
        return self.actions[action_id]
    
    def get_replay(self, size):
        local_batch = random.sample(self.memory, size)
        train = Training()
        for state, action_id, reward, next_state, status\
            in local_batch:
            output = self.model.predict(np.expand_dims(state, axis=0))[0]
            #If finished state do not predict
            if status:
                output[action_id] = reward
            else:
            #Otherwise predict relative to output model
                fit_output = self.output_model.predict(\
                        np.expand_dims(next_state, axis=0))[0]
                output[action_id] = reward + self.discount_rate * np.amax(fit_output)
            train.state.append(state)
            train.target.append(output)
        #Fit the initial values with the given states
        self.model.fit(np.array(train.state), np.array(train.target),\
                       epochs=1)
        self.decrease_epsilon()

    def decrease_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decrease
        elif self.epsilon != self.epsilon_min:
            self.epsilon = self.epsilon_min

    def memorize(self, state, action, reward, next_state, status):
        self.memory.append(\
            (state, self.actions.index(action), reward, next_state, status))

    def save(self, name):
        self.output_model.save_weights(name)

    def load(self, name):
        self.model.load_weights(name)
        self.update_output()