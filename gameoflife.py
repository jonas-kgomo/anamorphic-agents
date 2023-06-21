# An agent in the Game of Life by Jonas Kgomo
# Life is hard for AI 
# https://arxiv.org/abs/2009.01398
# Jacob M. Springer
# Garrett T. Kenyon
 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Define the Game of Life environment
class GameOfLife:
    def __init__(self, size):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        
    def update(self):
        new_grid = np.copy(self.grid)
        for i in range(self.size):
            for j in range(self.size):
                alive_neighbors = np.sum(self.grid[max(0, i-1):min(i+2, self.size), max(0, j-1):min(j+2, self.size)]) - self.grid[i, j]
                
                if self.grid[i, j] == 1:
                    if alive_neighbors < 2 or alive_neighbors > 3:
                        new_grid[i, j] = 0
                else:
                    if alive_neighbors == 3:
                        new_grid[i, j] = 1
                        
        self.grid = new_grid
        
    def reset(self):
        self.grid = np.zeros((self.size, self.size), dtype=int)
        
    def get_state(self):
        return self.grid
    
    def step(self, action):
        self.update()
        state = self.get_state()
        reward = np.sum(state)
        done = np.count_nonzero(state) == 0
        return state, reward, done

# Define the RL Agent
class RLAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Create the neural network model
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(state_size,)),
            tf.keras.layers.Dense(action_size)
        ])
        
        # Compile the model
        self.model.compile(optimizer='adam', loss='mse')
        
    def choose_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_size)
        else:
            q_values = self.model.predict(state.reshape(1, -1))
            return np.argmax(q_values[0])

    def update_q_values(self, states, actions, rewards, next_states, alpha, gamma):
        states = np.array(states).reshape(-1, self.state_size)
        next_states = np.array(next_states).reshape(-1, self.state_size)

        q_values = self.model.predict(states)
        next_q_values = self.model.predict(next_states)

        for i in range(len(states)):
            q_values[i][actions[i]] = q_values[i][actions[i]] + alpha * (
                rewards[i] + gamma * np.max(next_q_values[i]) - q_values[i][actions[i]]
            )

        self.model.fit(states, q_values, epochs=1, verbose=0, batch_size=len(states))

# Define the parameters
size = 10  # Size of the Game of Life grid
state_size = size * size
action_size = 4  # 4 possible actions: up, down, left, right
episodes = 1000
max_steps = 100
epsilon = 1.0  # Exploration rate
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor

# Create the environment and agent
env = GameOfLife(size)
agent = RLAgent(state_size, action_size)

# Training loop
rewards = []
for episode in range(episodes):
    state = env.get_state()
    total_reward = 0
    
    for step in range(max_steps):
        action = agent.choose_action(state.flatten(), epsilon)
        next_state, reward, done = env.step(action)
        
        agent.update_q_values(state.flatten(), [action], [reward], next_state.flatten(), alpha, gamma)
        
        state = next_state
        total_reward += reward
        
        if done:
            env.reset()
            break
    
    epsilon *= 0.99  # Decay epsilon over time
    rewards.append(total_reward)

# Plot the rewards over episodes
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Performance')
plt.show()
