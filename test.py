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
grid_config = GridConfig(map=grid, num_agents=10,observation_type="POMAPF",max_episode_steps=256, obs_radius=6)

# Create custom Pogema environment
env = pogema_v0(grid_config=grid_config)

anim_dir = str(pathlib.Path('renders') / "test")
env = AnimationMonitor(env, AnimationConfig(directory=anim_dir,egocentric_idx=0, static=False))
obs, info = env.reset()

while True:
    # Using random policy to make actions
    actions = env.sample_actions()
    # print(actions)
    obs, reward, terminated, truncated, info = env.step(actions)
    # env.render()
    if all(terminated) or all(truncated):
        print("test")
        break
