import numpy as np
import matplotlib.pyplot as plt

# State Space Deformation
def deform_state_space(state):
    # Apply deformation to the state space
    # Example: Modify the coordinates of the state
    deformed_state = (state[0] + 1, state[1] - 1)
    return deformed_state

# Function to plot the maze
def plot_maze(maze):
    plt.imshow(maze, cmap='gray')
    plt.axis('off')
    plt.show()

# Initial maze configuration
maze = np.zeros((10, 10))

# Define the starting and goal states
start_state = (5, 5)
goal_state = (8, 8)

# Plot the initial maze
plot_maze(maze)

# First Generation Deformation: State Space Deformation
deformed_start_state = deform_state_space(start_state)
deformed_goal_state = deform_state_space(goal_state)

deformed_maze = np.copy(maze)
deformed_maze[deformed_start_state] = 1
deformed_maze[deformed_goal_state] = 2

# Plot the maze after state space deformation
plot_maze(deformed_maze)

# Second Generation Deformation: Action Space Deformation
deformed_action = "Right"

deformed_maze[deformed_start_state] = 3  # Mark the new action
plot_maze(deformed_maze)

# Third Generation Deformation: Reward Deformation
deformed_reward = 10

deformed_maze[deformed_goal_state] = deformed_reward  # Assign the deformed reward to the goal state
plot_maze(deformed_maze)

# Fourth Generation Deformation: Task Deformation
deformed_goal_state = (2, 2)

deformed_maze[deformed_goal_state] = 2  # Update the goal state
plot_maze(deformed_maze)
