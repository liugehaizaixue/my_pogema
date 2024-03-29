from pogema import pogema_v0, GridConfig
import pathlib
from typing import Optional

from pogema.animation import AnimationConfig,AnimationMonitor
grid = """
.....#.....
.....#.....
...........
.....#.....
.....#.....
#.####.....
.....###.##
.....#.....
.....#.....
...........
.....#.....
"""


import sys

# Function to get keyboard input for action
def get_keyboard_action():
    action = input("Enter action (up/down/left/right or q to quit): ").lower()
    if action == 'up':
        return [0]  # Assuming 0 represents up action
    elif action == 'left':
        return [1]  # Assuming 1 represents down action
    elif action == 'right':
        return [2]  # Assuming 2 represents left action
    elif action == 'wait':
        return [3]  # Assuming 3 represents right action
    elif action == 'q':
        sys.exit()  # Quit if 'q' is entered
    else:
        print("Invalid action!")
        return get_keyboard_action()

# Define new configuration with 8 randomly placed agents
grid_config = GridConfig(map=grid, num_agents=1,observation_type="POMAPF",max_episode_steps=255)

# Create custom Pogema environment
env = pogema_v0(grid_config=grid_config)

anim_dir = str(pathlib.Path('renders') / "test")
env = AnimationMonitor(env, AnimationConfig(directory=anim_dir,egocentric_idx=0, static=False))
obs, info = env.reset()

while True:
    # Using random policy to make actions
    env.render()
    action = get_keyboard_action()
    obs, reward, terminated, truncated, info = env.step(action)
    if all(terminated) or all(truncated):
        break
