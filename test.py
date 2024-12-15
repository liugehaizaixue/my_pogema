from pogema import pogema_v0, GridConfig
import pathlib
from typing import Optional

from pogema.animation import AnimationConfig,AnimationMonitor
grid = """
..........###############............#################..........
..........################...........#################..........
..........#################..........#################..........
..........#################..........#################..........
..........################..........#################...........
..........################..........##################..........
#..........#############..#........##.###############.#.........
..#......#..###########.....#.........###############........#..
.#........#.############............#..#############.#........#.
............#############...............###########...........#.
.............############....###.#........#######...........#...
...........##############...................#####...............
...........###############...................####...............
...........##############.....................####..............
............############......................###...............
..............#########.......................###...............
..............#######.......................####................
................####........................###.................
.................###.......................###..................
.................###.......................###..................
..................###......................####........#........
.................####....................#####.......#..........
................###.......................###...................
................##........................###....#..............
................####.............#........###...................
..................###...#.............#....###..................
..................####..#..#.......#..#.....##..#...........#...
.............#....##..................#..##......#..........#...
.................##..................#...##.....................
....#........#...##......................####........#..........
....#.......#.......#......#.............######.................
.....######.........#.......#..##.#.....###########.............
....................##...............###############.....#######
...................######............################....####...
.........#####.....#######.......###...##...###...######........
........####...#########.....######..........##.....###.........
......#####....######.....#####..............###..........#.....
......##.......####.......#####...............###...............
.######........###.........##.................###...........#...
#####..............##......................##...............#...
####...............##......................##.....#........##...
###.................##....#...............####.....#.....#......
....................###................#...####......###........
................#..###....#.....#.....#.....####................
.............###...####..............#......####................
...................###.....................####.................
..................####....................####..................
..................####.....................###..................
.................####.......................####................
................####.........................###................
................####.......................###..................
.................####......................###..................
.................####......................####.................
........#.......######..............#.......####..#.............
................#########...#...............####...........#....
...........#...##########..............#....####.....#..........
.........#.......########.....#......#.....######...............
.................########.................#######...............
............#...#########...............#.########.#............
...............##########...............#..#######..#...........
...............##########...............#..########.............
...........##############...............#...######..#...........
...........##############.................#########..#..........
............#############...............#.##########.#..........
"""

# Define new configuration with 8 randomly placed agents
grid_config = GridConfig(seed=0, map=grid, num_agents=64,observation_type="POMAPF",max_episode_steps=512, obs_radius=6,map_name="test", display_directions=True)
# Create custom Pogema environment
env = pogema_v0(grid_config=grid_config)

anim_dir = str(pathlib.Path('renders') / "test")
env = AnimationMonitor(env, AnimationConfig(directory=anim_dir,egocentric_idx=0, static=False))

import numpy as np
_max_agents = 0

obs, info = env.reset()
for item in obs:
    _max_agents = max(_max_agents, np.sum(item['agents'] == 1))

for i in range(grid_config.max_episode_steps):
    # Using random policy to make actions
    actions = env.sample_actions()
    # print(actions)
    obs, reward, terminated, truncated, info = env.step(actions)
    for item in obs:
        _max_agents = max(_max_agents, np.sum(item['agents'] == 1))
    # env.render()
    #print(f'step {i}')
    if all(terminated) or all(truncated):
        print("test")
        break


print(_max_agents)