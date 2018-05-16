from .Agent import Agent

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import tensorflow as tf

import numpy as np
import random


class DQNAgent(Agent):
    """
    Agent using Deep Q Networks, implemented using keras.

    """

    def __init__(self, state_shape, num_action, memory_size=1000, build_model_func=None, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001):
        """

        :param state_shape: Shape of the environment state given as input input
        :param num_action: number of possible actions
        :param memory_size: size of the replay memory
        :param build_model_func: a function which returns a compiled keras model
        :param gamma: discount factor
        :param epsilon: exploration rate
        :param epsilon_min: min exploration
        :param epsilon_decay: rate of decay of the exploration rate
        :param learning_rate: learning rate for training the model
        """

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.learning_rate = learning_rate

        if build_model_func is not None and callable(build_model_func):
            self.__build_model__ = build_model_func

        super().__init__(state_shape, num_action, memory_size=memory_size)

    def __build_model__(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_shape, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.num_action, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.num_action)
        state = state.reshape((1,) + state.shape)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def learn(self, batch_size):
        if batch_size > len(self.memory):
            batch_size = len(self.memory)
        mini_batch = random.sample(self.memory, batch_size)
        states = []
        targets = []
        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            states.append(state)
            targets.append(target_f)
        states = np.squeeze(np.array(states))
        targets = np.squeeze(np.array(targets))
        self.model.fit(states, targets, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def remember(self, state, action, reward, next_state, done):
        state = state.reshape((1,) + state.shape)
        next_state = next_state.reshape((1,) + next_state.shape)
        super().remember(state, action, reward, next_state, done)


class DQNAgentTF(Agent):

    def __init__(self, state_shape, num_action, memory_size=1000, build_model_func=None, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.learning_rate = learning_rate

        self.inputs = tf.placeholder('float', shape=[None, state_shape])
        self.outputs = tf.placeholder('float', shape=[None, num_action])

        if build_model_func is not None and callable(build_model_func):
            self.__build_model__ = build_model_func

        super().__init__(state_shape, num_action, memory_size=memory_size)

    def __build_model__(self):
        layer1_weights = tf.Variable(initial_value=tf.random_normal([self.state_shape, 24]))
        layer1_bias = tf.Variable(initial_value=tf.random_normal([24]))
        layer1_outputs = tf.nn.relu(tf.add(tf.matmul(self.inputs, layer1_weights), layer1_bias))
        layer2_weights = tf.Variable(initial_value=tf.random_normal([24, 24]))
        layer2_bias = tf.Variable(initial_value=tf.random_normal([24]))
        layer2_outputs = tf.nn.relu(tf.add(tf.matmul(layer1_outputs, layer2_weights), layer2_bias))
        layer3_weights = tf.Variable(initial_value=tf.random_normal([24, self.num_action]))
        layer3_bias = tf.Variable(initial_value=tf.random_normal([self.num_action]))
        model_outputs = tf.nn.relu(tf.add(tf.matmul(layer2_outputs, layer3_weights), layer3_bias))
        return model_outputs

    def learn(self, batch_size):
        raise NotImplementedError

    def remember(self, state, action, reward, next_state, done):
        state = state.reshape((1,) + state.shape)
        next_state = next_state.reshape((1,) + next_state.shape)
        super().remember(state, action, reward, next_state, done)


