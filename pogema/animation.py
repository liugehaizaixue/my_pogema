import os
import typing
from itertools import cycle
from gymnasium import logger, Wrapper
import math
from pydantic import BaseModel
import numpy as np
from pogema import GridConfig, pogema_v0
from pogema.grid import Grid
from pogema.wrappers.persistence import PersistentWrapper, AgentState


class AnimationSettings(BaseModel):
    """
    Settings for the animation.
    """
    r: int = 35
    stroke_width: int = 10
    scale_size: int = 100
    time_scale: float = 0.5 # 0.28
    draw_start: int = 100
    rx: int = 15

    obstacle_color: str = '#84A1AE'
    ego_color: str = '#c1433c'
    ego_other_color: str = '#72D5C8'
    shaded_opacity: float = 0.2
    egocentric_shaded: bool = True
    stroke_dasharray: int = 25

    colors: list = [
        '#c1433c',
        '#2e6f9e',
        '#6e81af',
        '#00b9c8',
        '#72D5C8',
        '#0ea08c',
        '#8F7B66',
    ]


class AnimationConfig(BaseModel):
    """
    Configuration for the animation.
    """
    directory: str = 'renders/'
    static: bool = False
    show_agents: bool = True
    egocentric_idx: typing.Optional[int] = None
    uid: typing.Optional[str] = None
    save_every_idx_episode: typing.Optional[int] = 1
    show_border: bool = True
    show_lines: bool = False


class GridHolder(BaseModel):
    """
    Holds the grid and the history.
    """
    obstacles: typing.Any = None
    episode_length: int = None
    height: int = None
    width: int = None
    colors: dict = None
    history: list = None


class SvgObject:
    """
    Main class for the SVG.
    """
    tag = None

    def __init__(self, **kwargs):
        self.attributes = kwargs
        self.animations = []

    def add_animation(self, animation):
        self.animations.append(animation)

    @staticmethod
    def render_attributes(attributes):
        result = " ".join([f'{x.replace("_", "-")}="{y}"' for x, y in sorted(attributes.items())])
        return result

    def render(self):
        animations = '\n'.join([a.render() for a in self.animations]) if self.animations else None
        if animations:
            return f"<{self.tag} {self.render_attributes(self.attributes)}> {animations} </{self.tag}>"
        return f"<{self.tag} {self.render_attributes(self.attributes)} />"



class Sector(SvgObject):
    """
    Sector class for the SVG.
    """
    tag = 'path'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class Rectangle(SvgObject):
    """
    Rectangle class for the SVG.
    """
    tag = 'rect'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attributes['y'] = -self.attributes['y'] - self.attributes['height']


class Circle(SvgObject):
    """
    Circle class for the SVG.
    """
    tag = 'circle'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attributes['cy'] = -self.attributes['cy']


class Line(SvgObject):
    """
    Line class for the SVG.
    """
    tag = 'line'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attributes['y1'] = -self.attributes['y1']
        self.attributes['y2'] = -self.attributes['y2']


class Animation(SvgObject):
    """
    Animation class for the SVG.
    """
    tag = 'animate'

    def render(self):
        return f"<{self.tag} {self.render_attributes(self.attributes)}/>"


class Drawing:
    """
    Drawing, analog of the DrawSvg class in the pogema package.
    """

    def __init__(self, height, width, display_inline=False, origin=(0, 0)):
        self.height = height
        self.width = width
        self.display_inline = display_inline
        self.origin = origin
        self.elements = []

    def add_element(self, element):
        self.elements.append(element)

    def render(self):
        view_box = (0, -self.height, self.width, self.height)
        results = [f'''<?xml version="1.0" encoding="UTF-8"?>
        <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
             width="{self.width // 10}" height="{self.height // 10}" viewBox="{" ".join(map(str, view_box))}">''',
                   '\n<defs>\n', '</defs>\n']
        for element in self.elements:
            results.append(element.render())
        results.append('</svg>')
        return "\n".join(results)


class AnimationMonitor(Wrapper):
    """
    Defines the animation, which saves the episode as SVG.
    """

    def __init__(self, env, animation_config=AnimationConfig()):
        # Wrapping env using PersistenceWrapper for saving the history.
        env = PersistentWrapper(env)
        super().__init__(env)

        self.history = self.env.get_history()

        self.svg_settings: AnimationSettings = AnimationSettings()
        self.animation_config: AnimationConfig = animation_config

        self._episode_idx = 0

    def step(self, action):
        """
        Saves information about the episode.
        :param action: current actions
        :return: obs, reward, done, info
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        multi_agent_terminated = isinstance(terminated, (list, tuple)) and all(terminated)
        single_agent_terminated = isinstance(terminated, (bool, int)) and terminated
        multi_agent_truncated = isinstance(truncated, (list, tuple)) and all(truncated)
        single_agent_truncated = isinstance(truncated, (bool, int)) and truncated

        if multi_agent_terminated or single_agent_terminated or multi_agent_truncated or single_agent_truncated:
            save_tau = self.animation_config.save_every_idx_episode
            if save_tau:
                if (self._episode_idx + 1) % save_tau or save_tau == 1:
                    if not os.path.exists(self.animation_config.directory):
                        logger.info(f"Creating pogema monitor directory {self.animation_config.directory}", )
                        os.makedirs(self.animation_config.directory, exist_ok=True)

                    path = os.path.join(self.animation_config.directory,
                                        self.pick_name(self.grid_config, self._episode_idx))
                    self.save_animation(path)

        return obs, reward, terminated, truncated, info

    @staticmethod
    def pick_name(grid_config: GridConfig, episode_idx=None, zfill_ep=5):
        """
        Picks a name for the SVG file.
        :param grid_config: configuration of the grid
        :param episode_idx: idx of the episode
        :param zfill_ep: zfill for the episode number
        :return:
        """
        gc = grid_config
        name = 'pogema'
        if episode_idx is not None:
            name += f'-ep{str(episode_idx).zfill(zfill_ep)}'
        if gc:
            if gc.map_name:
                name += f'-{gc.map_name}'
            if gc.seed is not None:
                name += f'-seed{gc.seed}'
        else:
            name += '-render'
        return name + '.svg'

    def reset(self, **kwargs):
        """
        Resets the environment and resets the current positions of agents and targets
        :param kwargs:
        :return: obs: observation
        """
        obs = self.env.reset(**kwargs)

        self._episode_idx += 1
        self.history = self.env.get_history()

        return obs

    def create_animation(self, animation_config=None):
        """
        Creates the animation.
        :param animation_config: configuration of the animation
        :return: drawing: drawing object
        """
        anim_cfg = animation_config
        if anim_cfg is None:
            anim_cfg = self.animation_config

        grid: Grid = self.grid
        cfg = self.svg_settings
        colors = cycle(cfg.colors)
        agents_colors = {index: next(colors) for index in range(self.grid_config.num_agents)}

        if anim_cfg.egocentric_idx is not None:
            anim_cfg.egocentric_idx %= self.grid_config.num_agents

        decompressed_history: list[list[AgentState]] = self.env.decompress_history(self.history)

        # Change episode length for egocentric environment
        if anim_cfg.egocentric_idx is not None:
            episode_length = decompressed_history[anim_cfg.egocentric_idx][-1].step + 1
            for agent_idx in range(self.grid_config.num_agents):
                decompressed_history[agent_idx] = decompressed_history[agent_idx][:episode_length]
        else:
            episode_length = len(decompressed_history[0])

        # Add last observation one more time to highlight the final state
        for agent_idx in range(self.grid_config.num_agents):
            decompressed_history[agent_idx].append(decompressed_history[agent_idx][-1])

        # Change episode length for static environment
        if anim_cfg.static:
            episode_length = 1
            decompressed_history = [[decompressed_history[idx][-1]] for idx in range(len(decompressed_history))]

        gh = GridHolder(width=len(grid.obstacles), height=len(grid.obstacles[0]),
                        obstacles=grid.obstacles,
                        colors=agents_colors,
                        episode_length=episode_length,
                        history=decompressed_history, )

        render_width, render_height = gh.height * cfg.scale_size + cfg.scale_size, gh.width * cfg.scale_size + cfg.scale_size

        drawing = Drawing(width=render_width, height=render_height, display_inline=False, origin=(0, 0))
        obstacles = self.create_obstacles(gh, anim_cfg)

        agents = []
        targets = []

        if anim_cfg.show_agents:
            agents = self.create_agents(gh, anim_cfg)
            targets = self.create_targets(gh, anim_cfg)
            agents_direction = self.create_agents_direction(gh, anim_cfg)
            if not anim_cfg.static:
                self.animate_agents(agents, anim_cfg.egocentric_idx, gh)
                self.animate_agents_direction(agents_direction, anim_cfg.egocentric_idx, gh)
                self.animate_targets(targets, gh, anim_cfg)
        if anim_cfg.show_lines:
            grid_lines = self.create_grid_lines(gh, anim_cfg, render_width, render_height)
            for line in grid_lines:
                drawing.add_element(line)
        for obj in [*obstacles, *agents, *targets, *agents_direction,]:
            drawing.add_element(obj)

        if anim_cfg.egocentric_idx is not None:
            # field_of_view = self.create_field_of_view(grid_holder=gh, animation_config=anim_cfg)
            field_of_view = self.create_sector_field_of_view(grid_holder=gh, animation_config=anim_cfg)
            if not anim_cfg.static:
                self.animate_obstacles(obstacles=obstacles, grid_holder=gh, animation_config=anim_cfg)
                # self.animate_field_of_view(field_of_view, anim_cfg.egocentric_idx, gh)
                self.animate_sector_field_of_view(field_of_view, anim_cfg.egocentric_idx, gh)
            drawing.add_element(field_of_view)

        return drawing

    def create_grid_lines(self, grid_holder: GridHolder, animation_config: AnimationConfig, render_width,
                          render_height):
        """
        Creates the grid lines.
        :param grid_holder: grid holder
        :param animation_config: animation configuration
        :return: grid_lines: list of grid lines
        """
        cfg = self.svg_settings
        grid_lines = []
        for i in range(-1, grid_holder.height + 1):
            # vertical lines
            x0 = x1 = i * cfg.scale_size + cfg.scale_size / 2
            y0 = 0
            y1 = render_height
            grid_lines.append(
                Line(x1=x0, y1=y0, x2=x1, y2=y1, stroke=cfg.obstacle_color, stroke_width=cfg.stroke_width // 1.5))
        for i in range(-1, grid_holder.width + 1):
            # continue
            # horizontal lines
            x0 = 0
            y0 = y1 = i * cfg.scale_size + cfg.scale_size / 2
            x1 = render_width
            grid_lines.append(
                Line(x1=x0, y1=y0, x2=x1, y2=y1, stroke=cfg.obstacle_color, stroke_width=cfg.stroke_width // 1.5))

        # for i in range(grid_holder.width):
        #     grid_lines.append(Line(start=(0, i * cfg.scale_size),
        #                            end=(grid_holder.height * cfg.scale_size, i * cfg.scale_size),
        #                            stroke=cfg.grid_color, stroke_width=cfg.grid_width))
        return grid_lines

    def save_animation(self, name='render.svg', animation_config: typing.Optional[AnimationConfig] = None):
        """
        Saves the animation.
        :param name: name of the file
        :param animation_config: animation configuration
        :return: None
        """
        animation = self.create_animation(animation_config)
        with open(name, "w") as f:
            f.write(animation.render())

    @staticmethod
    def fix_point(x, y, length):
        """
        Fixes the point to the grid.
        :param x: coordinate x
        :param y: coordinate y
        :param length: size of the grid
        :return: x, y: fixed coordinates
        """
        return length - y - 1, x

    @staticmethod
    def check_in_radius(x1, y1, x2, y2, r) -> bool:
        """
        Checks if the point is in the radius.
        :param x1: coordinate x1  目标点
        :param y1: coordinate y1  目标点
        :param x2: coordinate x2  原点
        :param y2: coordinate y2  原点
        :param r: radius
        :return:
        """
        return x2 - r <= x1 <= x2 + r and y2 - r <= y1 <= y2 + r

    @staticmethod
    def get_positions_by_step(step, gh):
        """ 获取某个step中所有机器人的位置 """
        positions  = np.zeros_like(gh.obstacles)
        for agent in gh.history:
            if agent[step].active:
                x = agent[step].x
                y = agent[step].y
                if positions[x][y] == 1:
                    raise ValueError("positions error")
                else:
                    positions[x][y] =1  
        return positions

    @staticmethod
    def get_obstacles_and_agents_matrix(positions, obstacles):
        """ 获取obstacles与agents组合形成的obs_matrix """
        return np.logical_or(positions, obstacles).astype(int)

    @staticmethod
    def check_in_new_radius(direction0,x0,y0, x1,y1, r):
        def check_in_angle_range(direction0,x0,y0, x1,y1):
            """
            判断某个点是否位于可视的角度范围 
            注意 坐标系原点在左上角，且竖着的是x,横着的是y， 即越靠上的点x越小
            因此先给x0,x1取负号， 再让y=x,x=y进行坐标系转换
            """
            x0 = - x0
            x1 = - x1
            x0 , y0 = y0, x0
            x1 , y1 = y1, x1
            def calculate_angle(x0, y0, x1, y1):
                # 计算点与基准方向的水平距离和垂直距离
                delta_x = x1 - x0  # 假设基准方向的起点是坐标系原点 (0, 0), 向右为基准方向
                delta_y = y1 - y0
                # 使用反三角函数计算角度（以弧度为单位）
                angle_rad = math.atan2(delta_y, delta_x)
                # 将弧度转换为度数
                angle_deg = math.degrees(angle_rad)
                # 将角度限制在 0 到 360 度之间（可选）
                angle_deg = angle_deg % 360
                return angle_deg
            
            angle_deg = calculate_angle(x0, y0, x1, y1)
            direction_mapping = {
                (45, 135): [0, 1],   # up
                (135, 225): [-1, 0], # left
                (225, 315): [0, -1], # down
                (315, 360): [1, 0],  # right
                (0, 45): [1, 0]      # right (360 degrees is equivalent to 0 degrees)
            }          
            flag = False
            for angle_range, direction in direction_mapping.items():
                if angle_range[0] <= angle_deg <= angle_range[1]:
                    if direction == direction0:
                        flag = True
                    else:
                        continue
            return flag
            
        def check_in_sector_radius(x0, y0, x1, y1, r):
            """
            Checks if the point is in the radius. 用直线距离来判断
            :param x1: coordinate x0  原点
            :param y1: coordinate y0  原点
            :param x2: coordinate x1  目标点
            :param y2: coordinate y1  目标点
            :param r: radius
            :return:
            """
            distance = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
            return distance <= r
        
        if x0 == x1 and y0 == y1:
            return True

        return check_in_sector_radius(x0, y0, x1, y1, r) and check_in_angle_range(direction0,x0,y0, x1,y1)

    @staticmethod
    def check_is_before_obstacles_or_agent(x0, y0, x1, y1, obs_matrix, visibility_record) -> bool:
        """ 
        判断某个点是否可见，障碍物后不可见
        obs_matrix是obstacles与agents融合后的障碍物矩阵
        """
        if (x1,y1) in visibility_record:
            if visibility_record[(x1,y1)] == 1: # 已知可见
                return True
            elif visibility_record[(x1,y1)] == 0: #已知不可见
                return False
            else:
                raise ValueError('visibility record error')
            
        _x0 = x0
        _y0 = y0
        obs_matrix[x0][y0] = 0

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        points = []
        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        points.pop(0) # 排除代理当前位置这个点
        points.pop() #排除最终目标点

        def check_for_target_point(x0,y0, x1,y1, obs_matrix, visibility_record):
            """ 判断是否有障碍，且需判断是否可见 """
            """ 如果是line则只需判断 这条路径上是否有障碍 """
            if x1 - x0 == 0 or y1 - y0 == 0: #"line"
                """ 如果位于一条直线，此时已经检查过这条路径上的可见性，因此该点可见 """
                pass
            """ 如果位于斜方向 那么需要判断其相邻两个是否可见，且不为障碍 """
            if x1 - x0 < 0 and y1 - y0 > 0 : # "up-right"
                """ 相邻两点同时为障碍时，无论他们是否可见，该点都不可见 """
                if obs_matrix[x1][y1-1] != 0 and obs_matrix[x1+1][y1] != 0: #不能 同时为障碍
                    visibility_record[(x1,y1)] = 0
                    return False
                visibility_point0 = AnimationMonitor.check_is_before_obstacles_or_agent(x0, y0, x1+1 , y1, obs_matrix, visibility_record)
                visibility_point1 = AnimationMonitor.check_is_before_obstacles_or_agent(x0, y0, x1 , y1-1, obs_matrix, visibility_record)
                if visibility_point0 and visibility_point1:
                    """ 如果两个点都可见，且少于一个障碍，则目标点可见 
                        此处之前以排除 两个都是障碍的情况
                        因此此处应该返回 true 即不返回False即可 最终统一返回true
                    """
                    pass
                elif visibility_point0 and not visibility_point1 :
                    """ 如果只有一个点可见，那么该点不能是障碍 """
                    if obs_matrix[x1+1][y1] != 0:
                        visibility_record[(x1,y1)] = 0
                        return False
                elif visibility_point1 and not visibility_point0:
                    """ 如果只有一个点可见，那么该点不能是障碍 """
                    if obs_matrix[x1][y1-1] != 0:
                        visibility_record[(x1,y1)] = 0
                        return False
                elif not visibility_point0 and not visibility_point1:
                    """ 相邻两个点都不可见 ，因此该点不可见 """
                    visibility_record[(x1,y1)] = 0
                    return False
            elif x1 - x0 > 0 and y1 - y0 > 0: #"down-right"
                """ 相邻两点同时为障碍时，无论他们是否可见，该点都不可见 """
                if obs_matrix[x1][y1-1] != 0 and obs_matrix[x1-1][y1] != 0: #不能 同时为障碍
                    visibility_record[(x1,y1)] = 0
                    return False
                visibility_point0 = AnimationMonitor.check_is_before_obstacles_or_agent(x0, y0, x1 , y1-1, obs_matrix, visibility_record)
                visibility_point1 = AnimationMonitor.check_is_before_obstacles_or_agent(x0, y0, x1-1 , y1, obs_matrix, visibility_record)
                if visibility_point0 and visibility_point1:
                    """ 如果两个点都可见，且少于一个障碍，则目标点可见 
                        此处之前以排除 两个都是障碍的情况
                        因此此处应该返回 true 即不返回False即可 最终统一返回true
                    """
                    pass
                elif visibility_point0 and not visibility_point1 :
                    """ 如果只有一个点可见，那么该点不能是障碍 """
                    if obs_matrix[x1][y1-1] != 0:
                        visibility_record[(x1,y1)] = 0
                        return False
                elif visibility_point1 and not visibility_point0:
                    """ 如果只有一个点可见，那么该点不能是障碍 """
                    if obs_matrix[x1-1][y1] != 0:
                        visibility_record[(x1,y1)] = 0
                        return False
                elif not visibility_point0 and not visibility_point1:
                    """ 相邻两个点都不可见 ，因此该点不可见 """
                    visibility_record[(x1,y1)] = 0
                    return False
            elif x1 - x0 < 0 and y1 - y0 < 0: # "up-left"
                """ 相邻两点同时为障碍时，无论他们是否可见，该点都不可见 """
                if obs_matrix[x1][y1+1] != 0 and obs_matrix[x1+1][y1] != 0: #不能 同时为障碍
                    visibility_record[(x1,y1)] = 0
                    return False
                visibility_point0 = AnimationMonitor.check_is_before_obstacles_or_agent(x0, y0, x1 , y1+1, obs_matrix, visibility_record)
                visibility_point1 = AnimationMonitor.check_is_before_obstacles_or_agent(x0, y0, x1+1 , y1, obs_matrix, visibility_record)
                if visibility_point0 and visibility_point1:
                    """ 如果两个点都可见，且少于一个障碍，则目标点可见 
                        此处之前以排除 两个都是障碍的情况
                        因此此处应该返回 true 即不返回False即可 最终统一返回true
                    """
                    pass
                elif visibility_point0 and not visibility_point1 :
                    """ 如果只有一个点可见，那么该点不能是障碍 """
                    if obs_matrix[x1][y1+1] != 0:
                        visibility_record[(x1,y1)] = 0
                        return False
                elif visibility_point1 and not visibility_point0:
                    """ 如果只有一个点可见，那么该点不能是障碍 """
                    if obs_matrix[x1+1][y1] != 0:
                        visibility_record[(x1,y1)] = 0
                        return False
                elif not visibility_point0 and not visibility_point1:
                    """ 相邻两个点都不可见 ，因此该点不可见 """
                    visibility_record[(x1,y1)] = 0
                    return False
            elif x1 - x0 > 0 and y1 - y0 < 0 : #"down-left":
                """ 相邻两点同时为障碍时，无论他们是否可见，该点都不可见 """
                if obs_matrix[x1][y1+1] != 0 and obs_matrix[x1-1][y1] != 0: #不能 同时为障碍
                    visibility_record[(x1,y1)] = 0
                    return False
                visibility_point0 = AnimationMonitor.check_is_before_obstacles_or_agent(x0, y0, x1 , y1+1, obs_matrix, visibility_record)
                visibility_point1 = AnimationMonitor.check_is_before_obstacles_or_agent(x0, y0, x1-1 , y1, obs_matrix, visibility_record)
                if visibility_point0 and visibility_point1:
                    """ 如果两个点都可见，且少于一个障碍，则目标点可见 
                        此处之前以排除 两个都是障碍的情况
                        因此此处应该返回 true 即不返回False即可 最终统一返回true
                    """
                    pass
                elif visibility_point0 and not visibility_point1 :
                    """ 如果只有一个点可见，那么该点不能是障碍 """
                    if obs_matrix[x1][y1+1] != 0:
                        visibility_record[(x1,y1)] = 0
                        return False
                elif visibility_point1 and not visibility_point0:
                    """ 如果只有一个点可见，那么该点不能是障碍 """
                    if obs_matrix[x1-1][y1] != 0:
                        visibility_record[(x1,y1)] = 0
                        return False
                elif not visibility_point0 and not visibility_point1:
                    """ 相邻两个点都不可见 ，因此该点不可见 """
                    visibility_record[(x1,y1)] = 0
                    return False
                              
            visibility_record[(x1,y1)] = 1
            return True
        
        flag = True
        for x,y in points:
            """ 处理起点到终点之间的路径点的可视性 """
            if obs_matrix[x][y] != 0:  #说明 两点间的路径上有障碍
                flag = False
                break
            if (x,y) in visibility_record:
                if visibility_record[(x,y)] == 1: # 已知可见
                    continue
                elif visibility_record[(x,y)] == 0: #已知不可见
                    flag = False
                    break
                else:
                    raise ValueError('visibility record error')
            else: #可见性仍未知
                if AnimationMonitor.check_is_before_obstacles_or_agent(_x0, _y0, x , y, obs_matrix, visibility_record):
                    visibility_record[(x,y)] = 1
                    continue
                else:
                    visibility_record[(x,y)] = 0
                    flag = False
                    break
   
        if flag == True:
            if check_for_target_point(_x0, _y0, x1,y1, obs_matrix, visibility_record):
                return True
        return False


    @staticmethod
    def check_in_real_view(direction0,x0,y0, x1,y1, r, obs_matrix, visibility_record):
        """ 检查某个点是否可见 
            在视角范围内
            且 不被遮挡
        """
        if not AnimationMonitor.check_in_new_radius(direction0,x0,y0, x1,y1, r):
            return False
        
        if AnimationMonitor.check_is_before_obstacles_or_agent(x0,y0,x1,y1, obs_matrix, visibility_record):
            return True
        else:
            return False
        

    def create_sector_field_of_view(self, grid_holder, animation_config):
        """
        Creates the sector field of view for the egocentric agent.
        :param grid_holder:
        :param animation_config:
        :return:
        """
        cfg = self.svg_settings
        gh: GridHolder = grid_holder
        ego_idx = animation_config.egocentric_idx
        x, y = gh.history[ego_idx][0].get_xy()
        cx = cfg.draw_start + y * cfg.scale_size
        cy = cfg.draw_start + (gh.width - x - 1) * cfg.scale_size

        dr = (self.grid_config.obs_radius + 1) * cfg.scale_size - cfg.stroke_width * 2


        direction = gh.history[ego_idx][0].get_direction()
        if direction == [0,1]: #"UP"
            visual_angle = (225,315)
        elif direction == [0,-1]: #"DOWN"
            visual_angle = (45,135)
        elif direction == [-1,0]: #"LEFT"
            visual_angle = (135,225)
        else:                     #"RIGHT"
            visual_angle = (-45,45)

        d = self.create_sector_data(cx, -cy,dr - cfg.r, visual_angle[0], visual_angle[1]) #此处cy是负
        result = Sector(d=d, 
                        stroke=cfg.ego_color, stroke_width=cfg.stroke_width,
                        fill='none',
                        stroke_dasharray=cfg.stroke_dasharray,)
        return result

    def create_field_of_view(self, grid_holder, animation_config):
        """
        Creates the field of view for the egocentric agent.
        :param grid_holder:
        :param animation_config:
        :return:
        """
        cfg = self.svg_settings
        gh: GridHolder = grid_holder
        ego_idx = animation_config.egocentric_idx
        x, y = gh.history[ego_idx][0].get_xy()
        cx = cfg.draw_start + y * cfg.scale_size
        cy = cfg.draw_start + (gh.width - x - 1) * cfg.scale_size

        dr = (self.grid_config.obs_radius + 1) * cfg.scale_size - cfg.stroke_width * 2
        result = Rectangle(x=cx - dr + cfg.r, y=cy - dr + cfg.r,
                           width=2 * dr - 2 * cfg.r, height=2 * dr - 2 * cfg.r,
                           stroke=cfg.ego_color, stroke_width=cfg.stroke_width,
                           fill='none',
                           rx=cfg.rx, stroke_dasharray=cfg.stroke_dasharray,
                           )

        return result

    def create_sector_data(self, cx,cy,r,start_angle,end_angle):
        start_x = cx + r * math.cos(math.radians(start_angle))
        start_y = cy + r * math.sin(math.radians(start_angle))
        end_x = cx + r * math.cos(math.radians(end_angle))
        end_y = cy + r * math.sin(math.radians(end_angle))

        large_arc_flag = "0" if end_angle - start_angle <= 180 else "1"
        sweep_flag = "1"  # Always draw the arc in positive angle direction

        path_data = f"M {cx},{cy} " \
                    f"L {start_x},{start_y} " \
                    f"A {r},{r} 0 {large_arc_flag} {sweep_flag} {end_x},{end_y} " \
                    f"Z"
        return path_data

    def animate_sector_field_of_view(self, view, agent_idx, grid_holder,start_angle=45,end_angle=135):
        """
        Animates the field of view.
        :param view:
        :param agent_idx:
        :param grid_holder:
        :return:
        """
        gh: GridHolder = grid_holder
        cfg = self.svg_settings
        d_path = []

        for state in gh.history[agent_idx]:
            x, y = state.get_xy()
            dr = (self.grid_config.obs_radius + 1) * cfg.scale_size - cfg.stroke_width * 2
            cx = cfg.draw_start + y * cfg.scale_size
            cy = -cfg.draw_start + -(gh.width - x - 1) * cfg.scale_size

            direction = state.get_direction()
            if direction == [0,1]: #"UP"
                visual_angle = (225,315)
            elif direction == [0,-1]: #"DOWN"
                visual_angle = (45,135)
            elif direction == [-1,0]: #"LEFT"
                visual_angle = (135,225)
            else:                     #"RIGHT"
                visual_angle = (-45,45)
            d = self.create_sector_data(cx, cy, dr - cfg.r, visual_angle[0], visual_angle[1]) #此处cy是正
            d_path.append(d)

        visibility = ['visible' if state.is_active() else 'hidden' for state in gh.history[agent_idx]]

        view.add_animation(self.compressed_anim('d', d_path, cfg.time_scale))
        view.add_animation(self.compressed_anim('visibility', visibility, cfg.time_scale))

    def animate_field_of_view(self, view, agent_idx, grid_holder):
        """
        Animates the field of view.
        :param view:
        :param agent_idx:
        :param grid_holder:
        :return:
        """
        gh: GridHolder = grid_holder
        cfg = self.svg_settings
        x_path = []
        y_path = []
        for state in gh.history[agent_idx]:
            x, y = state.get_xy()
            dr = (self.grid_config.obs_radius + 1) * cfg.scale_size - cfg.stroke_width * 2
            cx = cfg.draw_start + y * cfg.scale_size
            cy = -cfg.draw_start + -(gh.width - x - 1) * cfg.scale_size
            x_path.append(str(cx - dr + cfg.r))
            y_path.append(str(cy - dr + cfg.r))

        visibility = ['visible' if state.is_active() else 'hidden' for state in gh.history[agent_idx]]

        view.add_animation(self.compressed_anim('x', x_path, cfg.time_scale))
        view.add_animation(self.compressed_anim('y', y_path, cfg.time_scale))
        view.add_animation(self.compressed_anim('visibility', visibility, cfg.time_scale))

    def animate_agents(self, agents, egocentric_idx, grid_holder):
        """
        Animates the agents.
        :param agents:
        :param egocentric_idx:
        :param grid_holder:
        :return:
        """
        gh: GridHolder = grid_holder
        cfg = self.svg_settings
        for agent_idx, agent in enumerate(agents):
            x_path = []
            y_path = []
            opacity = []
            for t_step , agent_state in enumerate(gh.history[agent_idx]):
                x, y = agent_state.get_xy()

                x_path.append(str(cfg.draw_start + y * cfg.scale_size))
                y_path.append(str(-cfg.draw_start + -(gh.width - x - 1) * cfg.scale_size))

                if egocentric_idx is not None:
                    ego_agent_state = gh.history[egocentric_idx][t_step]
                    ego_x, ego_y = ego_agent_state.get_xy()
                    ego_direction = ego_agent_state.get_direction()
                    positions = self.get_positions_by_step(t_step, gh)
                    obs_matirx = self.get_obstacles_and_agents_matrix(positions, gh.obstacles)
                    visibility_record = {(ego_x,ego_y):1} # 此表仅记录某元素是否可见，有值为 1确定可见，0不可见 ，无值则尚未确定
                    if self.check_in_real_view(ego_direction,ego_x, ego_y, x, y, self.grid_config.obs_radius, obs_matirx, visibility_record):
                        opacity.append('1.0')
                    else:
                        opacity.append(str(cfg.shaded_opacity))

            visibility = ['visible' if state.is_active() else 'hidden' for state in gh.history[agent_idx]]

            agent.add_animation(self.compressed_anim('cy', y_path, cfg.time_scale))
            agent.add_animation(self.compressed_anim('cx', x_path, cfg.time_scale))
            agent.add_animation(self.compressed_anim('visibility', visibility, cfg.time_scale))
            if opacity:
                agent.add_animation(self.compressed_anim('opacity', opacity, cfg.time_scale))

    def animate_agents_direction(self, agents_direction, egocentric_idx, grid_holder):
        """
        Animates the agents_direction.
        """
        gh: GridHolder = grid_holder
        cfg = self.svg_settings
        for agent_direction_idx, agent_direction in enumerate(agents_direction):
            x1_path = []
            y1_path = []
            x2_path = []
            y2_path = []
            opacity = []
            for t_step , agent_state in enumerate(gh.history[agent_direction_idx]):
                x, y = agent_state.get_xy()
                # 判断角度

                direction = agent_state.get_direction()
                r = cfg.r
                offset = (direction[0]*r , -direction[1]*r)
                x1_path.append(str(cfg.draw_start + y * cfg.scale_size))
                y1_path.append(str(-cfg.draw_start + -(gh.width - x - 1) * cfg.scale_size))
                x2_path.append(str(cfg.draw_start + y * cfg.scale_size + offset[0]))
                y2_path.append(str(-cfg.draw_start + -(gh.width - x - 1) * cfg.scale_size + offset[1]))  # 此处+ cfg.r 方向向下 

                if egocentric_idx is not None:
                    ego_agent_state = gh.history[egocentric_idx][t_step]
                    ego_x, ego_y = ego_agent_state.get_xy()
                    ego_direction = ego_agent_state.get_direction()
                    positions = self.get_positions_by_step(t_step, gh)
                    obs_matirx = self.get_obstacles_and_agents_matrix(positions, gh.obstacles)
                    visibility_record = {(ego_x,ego_y):1} # 此表仅记录某元素是否可见，有值为 1确定可见，0不可见 ，无值则尚未确定
                    if self.check_in_real_view(ego_direction,ego_x, ego_y, x, y, self.grid_config.obs_radius, obs_matirx, visibility_record):
                        opacity.append('1.0')
                    else:
                        opacity.append(str(cfg.shaded_opacity))

            visibility = ['visible' if state.is_active() else 'hidden' for state in gh.history[agent_direction_idx]]

            agent_direction.add_animation(self.compressed_anim('x1', x1_path, cfg.time_scale))
            agent_direction.add_animation(self.compressed_anim('y1', y1_path, cfg.time_scale))
            agent_direction.add_animation(self.compressed_anim('x2', x2_path, cfg.time_scale))
            agent_direction.add_animation(self.compressed_anim('y2', y2_path, cfg.time_scale))
            agent_direction.add_animation(self.compressed_anim('visibility', visibility, cfg.time_scale))
            if opacity:
                agent_direction.add_animation(self.compressed_anim('opacity', opacity, cfg.time_scale))

    @classmethod
    def compressed_anim(cls, attr_name, tokens, time_scale, rep_cnt='indefinite'):
        """
        Compresses the animation.
        :param attr_name:
        :param tokens:
        :param time_scale:
        :param rep_cnt:
        :return:
        """
        tokens, times = cls.compress_tokens(tokens)
        cumulative = [0, ]
        for t in times:
            cumulative.append(cumulative[-1] + t)
        times = [str(round(value / cumulative[-1], 10)) for value in cumulative]
        tokens = [tokens[0]] + tokens

        times = times
        tokens = tokens
        return Animation(attributeName=attr_name,
                         dur=f'{time_scale * (-1 + cumulative[-1])}s',
                         values=";".join(tokens),
                         repeatCount=rep_cnt,
                         keyTimes=";".join(times))

    @staticmethod
    def wisely_add(token, cnt, tokens, times):
        """
        Adds the token to the tokens and times.
        :param token:
        :param cnt:
        :param tokens:
        :param times:
        :return:
        """
        if cnt > 1:
            tokens += [token, token]
            times += [1, cnt - 1]
        else:
            tokens.append(token)
            times.append(cnt)

    @classmethod
    def compress_tokens(cls, input_tokens: list):
        """
        Compresses the tokens.
        :param input_tokens:
        :return:
        """
        tokens = []
        times = []
        if input_tokens:
            cur_idx = 0
            cnt = 1
            for idx in range(1, len(input_tokens)):
                if input_tokens[idx] == input_tokens[cur_idx]:
                    cnt += 1
                else:
                    cls.wisely_add(input_tokens[cur_idx], cnt, tokens, times)
                    cnt = 1
                    cur_idx = idx
            cls.wisely_add(input_tokens[cur_idx], cnt, tokens, times)
        return tokens, times

    def animate_targets(self, targets, grid_holder, animation_config):
        """
        Animates the targets.
        :param targets:
        :param grid_holder:
        :param animation_config:
        :return:
        """
        gh: GridHolder = grid_holder
        cfg = self.svg_settings
        ego_idx = animation_config.egocentric_idx

        for agent_idx, target in enumerate(targets):
            target_idx = ego_idx if ego_idx is not None else agent_idx

            x_path = []
            y_path = []

            for step_idx, state in enumerate(gh.history[target_idx]):
                x, y = state.get_target_xy()
                x_path.append(str(cfg.draw_start + y * cfg.scale_size))
                y_path.append(str(-cfg.draw_start + -(gh.width - x - 1) * cfg.scale_size))

            visibility = ['visible' if state.is_active() else 'hidden' for state in gh.history[agent_idx]]

            if self.grid_config.on_target == 'restart':
                target.add_animation(self.compressed_anim('cy', y_path, cfg.time_scale))
                target.add_animation(self.compressed_anim('cx', x_path, cfg.time_scale))
            target.add_animation(self.compressed_anim("visibility", visibility, cfg.time_scale))

    def create_obstacles(self, grid_holder, animation_config):
        """
        Creates the obstacles.
        :param grid_holder:
        :param animation_config:
        :return:
        """
        gh = grid_holder
        cfg = self.svg_settings

        result = []
        r = self.grid_config.obs_radius
        for i in range(gh.height):
            for j in range(gh.width):
                x, y = self.fix_point(i, j, gh.width)
                if not animation_config.show_border:
                    if i == r - 1 or j == r - 1 or j == gh.width - r or i == gh.height - r:
                        continue
                if gh.obstacles[x][y] != self.grid_config.FREE:
                    obs_settings = {}
                    obs_settings.update(x=cfg.draw_start + i * cfg.scale_size - cfg.r,
                                        y=cfg.draw_start + j * cfg.scale_size - cfg.r,
                                        width=cfg.r * 2,
                                        height=cfg.r * 2,
                                        rx=cfg.rx,
                                        fill=self.svg_settings.obstacle_color)

                    if animation_config.egocentric_idx is not None and cfg.egocentric_shaded:
                        initial_positions = [agent_states[0].get_xy() for agent_states in gh.history]
                        initial_directions = [agent_states[0].get_direction() for agent_states in gh.history]
                        ego_x, ego_y = initial_positions[animation_config.egocentric_idx]
                        ego_direction = initial_directions[animation_config.egocentric_idx]
                        positions = self.get_positions_by_step(0, gh)
                        obs_matirx = self.get_obstacles_and_agents_matrix(positions, gh.obstacles)
                        visibility_record = {(ego_x,ego_y):1} # 此表仅记录某元素是否可见，有值为 1确定可见，0不可见 ，无值则尚未确定
                        if not self.check_in_real_view(ego_direction,ego_x, ego_y, x, y, self.grid_config.obs_radius, obs_matirx, visibility_record):
                            obs_settings.update(opacity=cfg.shaded_opacity)

                    result.append(Rectangle(**obs_settings))

        return result

    def animate_obstacles(self, obstacles, grid_holder, animation_config):
        """

        :param obstacles:
        :param grid_holder:
        :param animation_config:
        :return:
        """
        gh: GridHolder = grid_holder
        obstacle_idx = 0
        cfg = self.svg_settings

        for i in range(gh.height):
            for j in range(gh.width):
                x, y = self.fix_point(i, j, gh.width)
                if gh.obstacles[x][y] == self.grid_config.FREE:
                    continue
                opacity = []
                seen = set()
                for t_step, agent_state in enumerate(gh.history[animation_config.egocentric_idx]):
                    ego_x, ego_y = agent_state.get_xy()
                    ego_direction = agent_state.get_direction()
                    positions = self.get_positions_by_step(t_step, gh)
                    obs_matirx = self.get_obstacles_and_agents_matrix(positions, gh.obstacles)
                    visibility_record = {(ego_x,ego_y):1} # 此表仅记录某元素是否可见，有值为 1确定可见，0不可见 ，无值则尚未确定
                    if self.check_in_real_view(ego_direction,ego_x, ego_y, x, y, self.grid_config.obs_radius, obs_matirx, visibility_record):
                        seen.add((x, y))
                    if (x, y) in seen:
                        opacity.append(str(1.0))
                    else:
                        opacity.append(str(cfg.shaded_opacity))

                obstacle = obstacles[obstacle_idx]
                obstacle.add_animation(self.compressed_anim('opacity', opacity, cfg.time_scale))

                obstacle_idx += 1

    def create_agents(self, grid_holder, animation_config):
        """
        Creates the agents.
        :param grid_holder:
        :param animation_config:
        :return:
        """
        gh: GridHolder = grid_holder
        cfg = self.svg_settings

        agents = []
        initial_positions = [agent_states[0].get_xy() for agent_states in gh.history]
        initial_directions = [agent_states[0].get_direction() for agent_states in gh.history]
        for idx, (x, y) in enumerate(initial_positions):

            if not any([agent_state.is_active() for agent_state in gh.history[idx]]):
                continue

            circle_settings = {}
            circle_settings.update(cx=cfg.draw_start + y * cfg.scale_size,
                                   cy=cfg.draw_start + (gh.width - x - 1) * cfg.scale_size,
                                   r=cfg.r, fill=gh.colors[idx])
            ego_idx = animation_config.egocentric_idx
            if ego_idx is not None:
                ego_x, ego_y = initial_positions[ego_idx]
                ego_direction = initial_directions[ego_idx]
                positions = self.get_positions_by_step(0, gh)
                obs_matirx = self.get_obstacles_and_agents_matrix(positions, gh.obstacles)
                visibility_record = {(ego_x,ego_y):1} # 此表仅记录某元素是否可见，有值为 1确定可见，0不可见 ，无值则尚未确定
                if not self.check_in_real_view(ego_direction,ego_x, ego_y, x, y, self.grid_config.obs_radius, obs_matirx, visibility_record) and cfg.egocentric_shaded:
                    circle_settings.update(opacity=cfg.shaded_opacity)
                if ego_idx == idx:
                    circle_settings.update(fill=self.svg_settings.ego_color)
                else:
                    circle_settings.update(fill=self.svg_settings.ego_other_color)
            agent = Circle(**circle_settings)
            agents.append(agent)

        return agents

    def create_agents_direction(self, grid_holder, animation_config):
        """
        Creates the agents.
        :param grid_holder:
        :param animation_config:
        :return:
        """
        gh: GridHolder = grid_holder
        cfg = self.svg_settings

        agents_direction = []
        initial_positions = [agent_states[0].get_xy() for agent_states in gh.history]
        initial_directions = [agent_states[0].get_direction() for agent_states in gh.history]
        for idx, (x, y) in enumerate(initial_positions):

            if not any([agent_state.is_active() for agent_state in gh.history[idx]]):
                continue
            cx=cfg.draw_start + y * cfg.scale_size
            cy=cfg.draw_start + (gh.width - x - 1) * cfg.scale_size
            r=cfg.r
            # 判断角度
            direction = initial_directions[idx]
            offset = (direction[0]*r , direction[1]*r)
            agent_direction = Line(x1=cx , y1=cy, x2=cx+ offset[0], y2=cy+ offset[1] , stroke="black" ) # 此处 -r 方向向下
            agents_direction.append(agent_direction)

        return agents_direction

    def create_targets(self, grid_holder, animation_config):
        """
        Creates the targets.
        :param grid_holder:
        :param animation_config:
        :return:
        """
        gh: GridHolder = grid_holder
        cfg = self.svg_settings
        targets = []
        for agent_idx, agent_states in enumerate(gh.history):

            tx, ty = agent_states[0].get_target_xy()
            x, y = ty, gh.width - tx - 1

            if not any([agent_state.is_active() for agent_state in gh.history[agent_idx]]):
                continue

            circle_settings = {}
            circle_settings.update(cx=cfg.draw_start + x * cfg.scale_size,
                                   cy=cfg.draw_start + y * cfg.scale_size,
                                   r=cfg.r,
                                   stroke=gh.colors[agent_idx], stroke_width=cfg.stroke_width, fill='none')
            if animation_config.egocentric_idx is not None:
                if animation_config.egocentric_idx != agent_idx:
                    continue

                circle_settings.update(stroke=cfg.ego_color)
            target = Circle(**circle_settings)
            targets.append(target)
        return targets


def main():
    grid_config = GridConfig(size=8, num_agents=5, obs_radius=2, seed=9, on_target='finish', max_episode_steps=128)
    env = pogema_v0(grid_config=grid_config)
    env = AnimationMonitor(env)

    env.reset()
    done = [False]

    while not all(done):
        _, _, done, _ = env.step(env.sample_actions())

    env.save_animation('out-static.svg', AnimationConfig(static=True, save_every_idx_episode=None))
    env.save_animation('out-static-ego.svg', AnimationConfig(egocentric_idx=0, static=True))
    env.save_animation('out-static-no-agents.svg', AnimationConfig(show_agents=False, static=True))
    env.save_animation("out.svg")
    env.save_animation("out-ego.svg", AnimationConfig(egocentric_idx=0))


if __name__ == '__main__':
    main()
