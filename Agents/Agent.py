from collections import deque


class Agent:

    def __init__(self, state_shape, num_action, memory_size=1000):
        self.state_shape = state_shape
        self.num_action = num_action
        self.model = self.__build_model__()
        self.memory = deque(maxlen=memory_size)

    def __build_model__(self):
        raise NotImplementedError

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, env_state):
        raise NotImplementedError

    def learn(self, batch_size):
        raise NotImplementedError



