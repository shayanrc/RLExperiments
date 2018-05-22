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
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            else:
                target = reward
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
        self.targets = tf.placeholder('float', shape=[None, num_action])

        self.session = tf.Session()

        if build_model_func is not None and callable(build_model_func):
            self.__build_model__ = build_model_func

        super().__init__(state_shape, num_action, memory_size=memory_size)

        self.model_action, self.model_outputs, self.optimizer = self.model

    def __build_model__(self):
        """
        Function to build a tensorflow model
        :return: A tuple containing the model's predicted action, output and the optimizer used for training the model
        """
        with tf.name_scope("Layer 1"):
            layer1_weights = tf.Variable(initial_value=tf.random_normal([self.state_shape, 24]))
            layer1_bias = tf.Variable(initial_value=tf.random_normal([24]))
            layer1_outputs = tf.nn.relu(tf.add(tf.matmul(self.inputs, layer1_weights), layer1_bias))
        with tf.name_scope("Layer 2"):
            layer2_weights = tf.Variable(initial_value=tf.random_normal([24, 24]))
            layer2_bias = tf.Variable(initial_value=tf.random_normal([24]))
            layer2_outputs = tf.nn.relu(tf.add(tf.matmul(layer1_outputs, layer2_weights), layer2_bias))
        with tf.name_scope("Layer 3"):
            layer3_weights = tf.Variable(initial_value=tf.random_normal([24, self.num_action]))
            layer3_bias = tf.Variable(initial_value=tf.random_normal([self.num_action]))
            model_outputs = tf.nn.relu(tf.add(tf.matmul(layer2_outputs, layer3_weights), layer3_bias))

        model_action = tf.argmax(model_outputs, axis=1, name="Action")
        model_loss = tf.losses.mean_squared_error(self.targets, model_outputs, name="Loss")
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(model_loss)
        self.session.run(tf.global_variables_initializer())
        return model_action, model_outputs, optimizer

    def act(self, state):
        """
        Get the model's action for the given state.
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.num_action)
        state = state.reshape((1,) + state.shape)
        action = self.session.run(self.model_action, feed_dict={self.inputs:state})
        return action[0]

    def learn(self, batch_size):
        if batch_size > len(self.memory):
            batch_size = len(self.memory)
        mini_batch = random.sample(self.memory, batch_size)
        states = []
        targets = []
        next_states = []
        for state, action, reward, next_state, done in mini_batch:
            if not done:
                output = self.session.run(self.model_outputs, feed_dict={self.inputs:next_state})
                target_val = reward + self.gamma * np.amax(output[0])
            else:
                target_val = reward
            target = self.session.run(self.model_outputs, feed_dict={self.inputs:state})            
            target[0][action] = target_val
            states.append(state)
            targets.append(target)
        states = np.squeeze(np.array(states))
        targets = np.squeeze(np.array(targets))
        self.session.run(self.optimizer, feed_dict={self.inputs:states, self.targets:targets})
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def remember(self, state, action, reward, next_state, done):
        """
        Save the state, action, reward, the next state and a boolean value indicating whether the episode ended here.
        """
        state = state.reshape((1,) + state.shape)
        next_state = next_state.reshape((1,) + next_state.shape)
        super().remember(state, action, reward, next_state, done)

    def __del__(self):
        self.session.close()


