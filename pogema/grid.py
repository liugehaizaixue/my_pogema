from copy import deepcopy
import warnings

import numpy as np
import math
from pogema.generator import generate_obstacles, generate_positions_and_targets_fast, \
    get_components
from .grid_config import GridConfig
from .grid_registry import in_registry, get_grid
from .utils import render_grid


class Grid:

    def __init__(self, grid_config: GridConfig, add_artificial_border: bool = True, num_retries=10):

        self.config = grid_config
        self.rnd = np.random.default_rng(grid_config.seed)

        if self.config.map is None:
            self.obstacles = generate_obstacles(self.config)
        else:
            self.obstacles = np.array([np.array(line) for line in self.config.map])
        if in_registry(self.config.map_name):
            self.obstacles = get_grid(self.config.map_name).get_obstacles()
        self.obstacles = self.obstacles.astype(np.int32)

        if grid_config.targets_xy and grid_config.agents_xy:
            self.starts_xy, self.finishes_xy = grid_config.agents_xy, grid_config.targets_xy
            if len(self.starts_xy) != len(self.finishes_xy):
                raise IndexError("Can't create task. Please provide agents_xy and targets_xy of the same size.")
            grid_config.num_agents = len(self.starts_xy)
            for start_xy, finish_xy in zip(self.starts_xy, self.finishes_xy):
                s_x, s_y = start_xy
                f_x, f_y = finish_xy
                if self.config.map is not None and self.obstacles[s_x, s_y] == grid_config.OBSTACLE:
                    warnings.warn(f"There is an obstacle on a start point ({s_x}, {s_y}), replacing with free cell",
                                  Warning, stacklevel=2)
                self.obstacles[s_x, s_y] = grid_config.FREE
                if self.config.map is not None and self.obstacles[f_x, f_y] == grid_config.OBSTACLE:
                    warnings.warn(f"There is an obstacle on a finish point ({s_x}, {s_y}), replacing with free cell",
                                  Warning, stacklevel=2)
                self.obstacles[f_x, f_y] = grid_config.FREE
        else:
            self.starts_xy, self.finishes_xy, self.starts_direction, self.finishes_direction = generate_positions_and_targets_fast(self.obstacles, self.config)

        if len(self.starts_xy) != len(self.finishes_xy):
            for attempt in range(num_retries):
                if len(self.starts_xy) == len(self.finishes_xy):
                    warnings.warn(f'Created valid configuration only with {attempt} attempts.', Warning, stacklevel=2)
                    break
                if self.config.map is None:
                    self.obstacles = generate_obstacles(self.config)
                self.starts_xy, self.finishes_xy, self.starts_direction, self.finishes_direction = generate_positions_and_targets_fast(self.obstacles, self.config)

        if not self.starts_xy or not self.finishes_xy or len(self.starts_xy) != len(self.finishes_xy):
            raise OverflowError(
                "Can't create task. Please check grid grid_config, especially density, num_agent and map.")

        if add_artificial_border:
            self.add_artificial_border()

        filled_positions = np.zeros(self.obstacles.shape)
        for x, y in self.starts_xy:
            filled_positions[x, y] = 1

        self.positions = filled_positions
        self.positions_direction = self.starts_direction
        self.positions_xy = self.starts_xy
        self._initial_xy = deepcopy(self.starts_xy)
        self.is_active = {agent_id: True for agent_id in range(self.config.num_agents)}

    def add_artificial_border(self):
        gc = self.config
        r = gc.obs_radius
        if gc.empty_outside:
            filled_obstacles = np.zeros(np.array(self.obstacles.shape) + r * 2)
        else:
            filled_obstacles = self.rnd.binomial(1, gc.density, np.array(self.obstacles.shape) + r * 2)

        height, width = filled_obstacles.shape
        filled_obstacles[r - 1, r - 1:width - r + 1] = gc.OBSTACLE
        filled_obstacles[r - 1:height - r + 1, r - 1] = gc.OBSTACLE
        filled_obstacles[height - r, r - 1:width - r + 1] = gc.OBSTACLE
        filled_obstacles[r - 1:height - r + 1, width - r] = gc.OBSTACLE
        filled_obstacles[r:height - r, r:width - r] = self.obstacles

        self.obstacles = filled_obstacles

        self.starts_xy = [(x + r, y + r) for x, y in self.starts_xy]
        self.finishes_xy = [(x + r, y + r) for x, y in self.finishes_xy]

    def get_obstacles(self, ignore_borders=False):
        gc = self.config
        if ignore_borders:
            return self.obstacles[gc.obs_radius:-gc.obs_radius, gc.obs_radius:-gc.obs_radius].copy()
        return self.obstacles.copy()

    @staticmethod
    def _cut_borders_xy(positions, obs_radius):
        return [[x - obs_radius, y - obs_radius] for x, y in positions]

    @staticmethod
    def _filter_inactive(pos, active_flags):
        return [pos for idx, pos in enumerate(pos) if active_flags[idx]]

    def get_grid_config(self):
        return deepcopy(self.config)

    # def _get_grid_config(self) -> GridConfig:
    #     return self.env.grid_config

    def _prepare_positions(self, positions, only_active, ignore_borders):
        gc = self.config

        if only_active:
            positions = self._filter_inactive(positions, [idx for idx, active in self.is_active.items() if active])

        if ignore_borders:
            positions = self._cut_borders_xy(positions, gc.obs_radius)

        return positions

    def get_agents_xy(self, only_active=False, ignore_borders=False):
        return self._prepare_positions(deepcopy(self.positions_xy), only_active, ignore_borders)

    @staticmethod
    def to_relative(coordinates, offset):
        result = deepcopy(coordinates)
        for idx, _ in enumerate(result):
            x, y = result[idx]
            dx, dy = offset[idx]
            result[idx] = x - dx, y - dy
        return result

    def get_agents_direction_relative(self):
        return self.positions_direction

    def get_agents_xy_relative(self):
        return self.to_relative(self.positions_xy, self._initial_xy)

    def get_targets_xy_relative(self):
        return self.to_relative(self.finishes_xy, self._initial_xy)

    def get_targets_xy(self, only_active=False, ignore_borders=False):
        return self._prepare_positions(deepcopy(self.finishes_xy), only_active, ignore_borders)

    def _normalize_coordinates(self, coordinates):
        gc = self.config

        x, y = coordinates

        x -= gc.obs_radius
        y -= gc.obs_radius

        x /= gc.size - 1
        y /= gc.size - 1

        return x, y

    def get_state(self, ignore_borders=False, as_dict=False):
        agents_xy = list(map(self._normalize_coordinates, self.get_agents_xy(ignore_borders)))
        targets_xy = list(map(self._normalize_coordinates, self.get_targets_xy(ignore_borders)))

        obstacles = self.get_obstacles(ignore_borders)

        if as_dict:
            return {"obstacles": obstacles, "agents_xy": agents_xy, "targets_xy": targets_xy}

        return np.concatenate(list(map(lambda x: np.array(x).flatten(), [agents_xy, targets_xy, obstacles])))

    def get_observation_shape(self):
        full_radius = self.config.obs_radius * 2 + 1
        return 2, full_radius, full_radius

    def get_num_actions(self):
        return len(self.config.MOVES)

    def get_obstacles_for_agent(self, agent_id):
        x, y = self.positions_xy[agent_id]
        r = self.config.obs_radius
        return self.obstacles[x - r:x + r + 1, y - r:y + r + 1].astype(np.float32)

    @staticmethod
    def check_is_before_obstacles_or_agent(x0, y0, x1, y1, obs_matrix, visibility_record_matrix) -> bool:
        """ 
        判断某个点是否可见，障碍物/agent之后的不可见
        obs_matrix是两者融合后的障碍物矩阵
        """
        n = len(obs_matrix[0])
        i = n-y1-1 # 行
        j = x1 # 列
        if visibility_record_matrix[i][j] == 1:
            return True
        elif visibility_record_matrix[i][j] == 0:
            return False

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

        def check_for_target_point(x0,y0, x1,y1, obs_matrix,visibility_record_matrix):
            """ 判断是否有障碍，且需判断是否可见 """
            """ 如果是line则只需判断，这条路径上是否有障碍 """
            n = len(obs_matrix[0])
            i = n-y1-1 # 行
            j = x1 # 列
            _type = ""
            if x1 - x0 == 0 or y1 - y0 == 0: #"line"
                """ 如果位于一条直线，此时已经检查过这条路径上的可见性，因此该点可见 """
                pass

            """ 如果位于斜方向 那么需要判断其相邻两个是否可见，且不为障碍 """
            if x1 - x0 > 0 and y1 - y0 > 0 : # "up-right"
                """ 相邻两点同时为障碍时，无论他们是否可见，该点都不可见 """
                if obs_matrix[i][j-1] != 0 and obs_matrix[i+1][j] != 0: #不能 同时为障碍
                    visibility_record_matrix[i][j] = 0
                    return False
                visibility_point0 = Grid.check_is_before_obstacles_or_agent(x0, y0, x1-1 , y1, obs_matrix, visibility_record_matrix)
                visibility_point1 = Grid.check_is_before_obstacles_or_agent(x0, y0, x1 , y1-1, obs_matrix, visibility_record_matrix)
                if visibility_point0 and visibility_point1:
                    """ 如果两个点都可见，且少于一个障碍，则目标点可见 
                        此处之前以排除 两个都是障碍的情况
                        因此此处应该返回 true，即不返回False即可，最终统一返回true
                    """
                    pass
                elif visibility_point0 and not visibility_point1 :
                    """ 如果只有一个点可见，那么该点不能是障碍 """
                    if obs_matrix[i][j-1] != 0:
                        visibility_record_matrix[i][j] = 0
                        return False
                elif visibility_point1 and not visibility_point0:
                    """ 如果只有一个点可见，那么该点不能是障碍 """
                    if obs_matrix[i+1][j] != 0:
                        visibility_record_matrix[i][j] = 0
                        return False
                elif not visibility_point0 and not visibility_point1:
                    """ 相邻两个点都不可见 ，因此该点不可见 """
                    visibility_record_matrix[i][j] = 0
                    return False  
            elif x1 - x0 > 0 and y1 - y0 < 0: #"down-right"
                if obs_matrix[i][j-1] != 0 and obs_matrix[i-1][j] != 0:
                    visibility_record_matrix[i][j] = 0
                    return False
                visibility_point0 = Grid.check_is_before_obstacles_or_agent(x0, y0, x1-1 , y1, obs_matrix, visibility_record_matrix)
                visibility_point1 = Grid.check_is_before_obstacles_or_agent(x0, y0, x1 , y1+1, obs_matrix, visibility_record_matrix)
                if visibility_point0 and visibility_point1:
                    """ 如果两个点都可见，且少于一个障碍，则目标点可见 
                        此处之前以排除 两个都是障碍的情况
                        因此此处应该返回 true，即不返回False即可，最终统一返回true
                    """
                    pass
                elif visibility_point0 and not visibility_point1 :
                    """ 如果只有一个点可见，那么该点不能是障碍 """
                    if obs_matrix[i][j-1] != 0:
                        visibility_record_matrix[i][j] = 0
                        return False
                elif visibility_point1 and not visibility_point0:
                    """ 如果只有一个点可见，那么该点不能是障碍 """
                    if obs_matrix[i-1][j] != 0:
                        visibility_record_matrix[i][j] = 0
                        return False
                elif not visibility_point0 and not visibility_point1:
                    """ 相邻两个点都不可见 ，因此该点不可见 """
                    visibility_record_matrix[i][j] = 0
                    return False  
            elif x1 - x0 < 0 and y1 - y0 > 0: # "up-left"
                if obs_matrix[i][j+1] != 0 and obs_matrix[i+1][j] != 0:
                    visibility_record_matrix[i][j] = 0
                    return False
                visibility_point0 = Grid.check_is_before_obstacles_or_agent(x0, y0, x1 , y1-1, obs_matrix, visibility_record_matrix)
                visibility_point1 = Grid.check_is_before_obstacles_or_agent(x0, y0, x1+1 , y1, obs_matrix, visibility_record_matrix)
                if visibility_point0 and visibility_point1:
                    """ 如果两个点都可见，且少于一个障碍，则目标点可见 
                        此处之前以排除 两个都是障碍的情况
                        因此此处应该返回 true，即不返回False即可，最终统一返回true
                    """
                    pass
                elif visibility_point0 and not visibility_point1 :
                    """ 如果只有一个点可见，那么该点不能是障碍 """
                    if obs_matrix[i+1][j] != 0:
                        visibility_record_matrix[i][j] = 0
                        return False
                elif visibility_point1 and not visibility_point0:
                    """ 如果只有一个点可见，那么该点不能是障碍 """
                    if obs_matrix[i][j+1] != 0:
                        visibility_record_matrix[i][j] = 0
                        return False
                elif not visibility_point0 and not visibility_point1:
                    """ 相邻两个点都不可见 ，因此该点不可见 """
                    visibility_record_matrix[i][j] = 0
                    return False 
            elif x1 - x0 < 0 and y1 - y0 < 0 : #"down-left":
                if obs_matrix[i][j+1] != 0 and obs_matrix[i-1][j] != 0:
                    visibility_record_matrix[i][j] = 0
                    return False
                visibility_point0 = Grid.check_is_before_obstacles_or_agent(x0, y0, x1 , y1+1, obs_matrix, visibility_record_matrix)
                visibility_point1 = Grid.check_is_before_obstacles_or_agent(x0, y0, x1+1 , y1, obs_matrix, visibility_record_matrix)
                if visibility_point0 and visibility_point1:
                    """ 如果两个点都可见，且少于一个障碍，则目标点可见 
                        此处之前以排除 两个都是障碍的情况
                        因此此处应该返回 true，即不返回False即可，最终统一返回true
                    """
                    pass
                elif visibility_point0 and not visibility_point1 :
                    """ 如果只有一个点可见，那么该点不能是障碍 """
                    if obs_matrix[i-1][j] != 0:
                        visibility_record_matrix[i][j] = 0
                        return False
                elif visibility_point1 and not visibility_point0:
                    """ 如果只有一个点可见，那么该点不能是障碍 """
                    if obs_matrix[i][j+1] != 0:
                        visibility_record_matrix[i][j] = 0
                        return False
                elif not visibility_point0 and not visibility_point1:
                    """ 相邻两个点都不可见 ，因此该点不可见 """
                    visibility_record_matrix[i][j] = 0
                    return False
            visibility_record_matrix[i][j] = 1
            return True

        flag = True
        for k in range(len(points)):
            """ 处理起点到终点之间的路径点的可视性 """
            n = len(obs_matrix[0])
            x, y = points[k]
            i = n-y-1 # 行
            j = x # 列
            if obs_matrix[i][j] != 0:  #说明 两点间的路径上有障碍
                flag = False
                break

            if visibility_record_matrix[i][j] == 0:
                flag = False
                break
            elif visibility_record_matrix[i][j] == 1:
                continue
            else:
                if Grid.check_is_before_obstacles_or_agent(_x0, _y0, x , y, obs_matrix, visibility_record_matrix):
                    visibility_record_matrix[i][j] = 1
                    continue
                else:
                    visibility_record_matrix[i][j] = 0
                    flag = False
                    break
        
        if flag == True:
            if check_for_target_point(_x0, _y0, x1,y1, obs_matrix, visibility_record_matrix):
                return True
        return False


    @staticmethod
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

    @staticmethod
    def check_in_sector_radius(x0, y0, x1, y1, r):
        """
        Checks if the point is in the radius. 用直线距离来判断,超出r的不可见
        :param x1: coordinate x0  原点
        :param y1: coordinate y0  原点
        :param x2: coordinate x1  目标点
        :param y2: coordinate y1  目标点
        :param r: radius
        :return:
        """
        distance = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        return distance <= r  

    @staticmethod
    def check_in_angle_range(angle_deg, direction0):
        """ 判断是否位于可视角度内 """
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

    @staticmethod
    def get_sector_range_from_rect(matrix0 ,direction):
        """ 
        从rect矩阵中裁取sector范围
        """
        n = len(matrix0)
        x0 = y0 =  (n - 1) // 2
        for i in range(n):
            # 遍历列
            for j in range(n):
                x1 = j
                y1 = n - i -1
                """ 左上角为坐标原点，进行坐标转换, 转换结果为左下角为0,0 
                    x向右为正，y向上为正
                """
                angle_deg = Grid.calculate_angle(x0,y0,x1,y1)
                if not Grid.check_in_angle_range(angle_deg,direction) and not (x0 == x1 and y0 == y1) :
                    matrix0[i][j] = -1 # 将不可见区域改为-1
                    continue
                if not Grid.check_in_sector_radius(x0,x0,x1,y1, r = x0):
                    matrix0[i][j] = -1 # 将不可见区域改为-1
                    continue
        return matrix0
    
    @staticmethod
    def get_real_views(matrix0 , matrix1, direction):
        """
        matrix 代表obstacles与positions两个矩阵
        matrix0 为最终返回的矩阵
        matrix1 为另一个矩阵，他们的组合会产生视觉影响
        不可见区域设为-1
        sector范围外的:角度范围外，直线距离外，障碍物后
        """
        obs_matrix = np.logical_or(matrix0 , matrix1).astype(int) # 通过布尔运算，产生obstacles+position组成的障碍图，根据该图判断 障碍物后的点是否可见
        obs_matrix_sector = Grid.get_sector_range_from_rect(obs_matrix, direction)
        matrix0_sector = Grid.get_sector_range_from_rect(deepcopy(matrix0), direction)
        visibility_record_matrix  = [[-1 if element != -1 else 0 for element in row] for row in obs_matrix_sector]  # 此表仅记录某元素是否可见，-1尚未判断，1确定可见，0不可见
        n = len(matrix0_sector)
        x0 = y0 = int(((1+n) / 2) -1)
        visibility_record_matrix[x0][y0] = 1
        for i in range(n):
            # 遍历列
            for j in range(len(matrix0_sector[i])):
                x1 = j
                y1 = n - i -1
                """ 左上角为坐标原点，进行坐标转换, 转换结果为左下角为0,0 """
                if visibility_record_matrix[i][j] == 1: # 已知可见
                    continue
                elif visibility_record_matrix[i][j] == 0: # 已知不可见
                    matrix0_sector[i][j] = -1 # 将不可见区域改为-1
                else: #未知
                    if not Grid.check_is_before_obstacles_or_agent(x0,y0,x1,y1,obs_matrix_sector, visibility_record_matrix):
                        matrix0_sector[i][j] = -1 # 将不可见区域改为-1
                        visibility_record_matrix[i][j] = 0
                        continue
                    else:
                        visibility_record_matrix[i][j] = 1
        return matrix0_sector

    def new_get_obstacles_for_agent(self, agent_id):
        x, y = self.positions_xy[agent_id]
        direction = self.positions_direction[agent_id]
        r = self.config.obs_radius
        rect_obstacles = self.obstacles[x - r:x + r + 1, y - r:y + r + 1]
        rect_positions = self.positions[x - r:x + r + 1, y - r:y + r + 1]
        sector_obstacles = self.get_real_views(rect_obstacles, rect_positions, direction)
        return sector_obstacles.astype(np.float32)

    def new_get_positions(self, agent_id):
        x, y = self.positions_xy[agent_id]
        direction = self.positions_direction[agent_id]
        r = self.config.obs_radius
        rect_obstacles = self.obstacles[x - r:x + r + 1, y - r:y + r + 1]
        rect_positions = self.positions[x - r:x + r + 1, y - r:y + r + 1]
        sector_positions = self.get_real_views(rect_positions, rect_obstacles, direction)
        if self.config.display_directions:
            other_positions = np.where(sector_positions == 1)
            other_positions_list = list(zip(other_positions[0]-r + x, other_positions[1]-r + y))
            other_agents_id_list = [i for i, x in enumerate(self.positions_xy) if x in other_positions_list]
            # 过滤不活跃的agent
            other_agents_id_list = [idx for idx in other_agents_id_list if self.is_active[idx]]
            other_agents_directions_list = [self.positions_direction[idx] for idx in other_agents_id_list]
            vector_mapping = {
                (1,0): 1,
                (0,-1): 2,
                (-1,0): 3,
                (0,1): 4
            }

            for k in range(len(other_agents_directions_list)):
                vector = other_agents_directions_list[k]
                direction_number = vector_mapping.get(tuple(vector), 0)  # 如果找不到对应的数字，返回默认值0
                if direction_number != 0:
                    # 将对应的数字更新到 sector_positions 中
                    i , j = other_positions_list[k][0] - x + r , other_positions_list[k][1] - y + r
                    if (i , j) in zip(*other_positions):
                        sector_positions[i, j] = direction_number
                    else:
                        raise ValueError("Invalid position: {}".format(other_positions_list[k]))
                else:
                    raise ValueError("Invalid vector: {}".format(vector))
            return sector_positions.astype(np.float32)
        else:
            return sector_positions.astype(np.float32)

    def get_positions(self, agent_id):
        x, y = self.positions_xy[agent_id]
        r = self.config.obs_radius
        return self.positions[x - r:x + r + 1, y - r:y + r + 1].astype(np.float32)
    
    def get_target(self, agent_id):

        x, y = self.positions_xy[agent_id]
        fx, fy = self.finishes_xy[agent_id]
        if x == fx and y == fy:
            return 0.0, 0.0
        rx, ry = fx - x, fy - y
        dist = np.sqrt(rx ** 2 + ry ** 2)
        return rx / dist, ry / dist

    def get_square_target(self, agent_id):
        c = self.config
        full_size = self.config.obs_radius * 2 + 1
        result = np.zeros((full_size, full_size))
        x, y = self.positions_xy[agent_id]
        fx, fy = self.finishes_xy[agent_id]
        dx, dy = x - fx, y - fy

        dx = min(dx, c.obs_radius) if dx >= 0 else max(dx, -c.obs_radius)
        dy = min(dy, c.obs_radius) if dy >= 0 else max(dy, -c.obs_radius)
        result[c.obs_radius - dx, c.obs_radius - dy] = 1
        return result.astype(np.float32)

    def render(self, mode='human'):
        render_grid(self.obstacles, self.positions_xy, self.finishes_xy, self.is_active, mode=mode)

    def move_agent_to_cell(self, agent_id, x, y):
        if self.positions[self.positions_xy[agent_id]] == self.config.FREE:
            raise KeyError("Agent {} is not in the map".format(agent_id))
        self.positions[self.positions_xy[agent_id]] = self.config.FREE
        if self.obstacles[x, y] != self.config.FREE or self.positions[x, y] != self.config.FREE:
            raise ValueError(f"Can't force agent to blocked position {x} {y}")
        self.positions_xy[agent_id] = x, y
        self.positions[self.positions_xy[agent_id]] = self.config.OBSTACLE

    def has_obstacle(self, x, y):
        return self.obstacles[x, y] == self.config.OBSTACLE

    def move_without_checks(self, agent_id, action):
        x, y = self.positions_xy[agent_id]
        dx, dy = self.config.MOVES[action]
        self.positions[x, y] = self.config.FREE
        self.positions[x+dx, y+dy] = self.config.OBSTACLE
        self.positions_xy[agent_id] = (x+dx, y+dy)

    def move(self, agent_id, action):
        x, y = self.positions_xy[agent_id]
        dx, dy = self.config.MOVES[action]
        if self.obstacles[x + dx, y + dy] == self.config.FREE:
            if self.positions[x + dx, y + dy] == self.config.FREE:
                self.positions[x, y] = self.config.FREE
                x += dx
                y += dy
                self.positions[x, y] = self.config.OBSTACLE
        self.positions_xy[agent_id] = (x, y)

    def new_move(self, agent_id, action):
        x, y = self.positions_xy[agent_id]
        direction = self.positions_direction[agent_id]
        action = self.config.NEW_MOVES[action]
        # print(self.positions_xy[agent_id])
        # print(self.positions_direction[agent_id])
        # print(action)
        if action == "TURN_LEFT":
            direction = [-direction[1], direction[0]]
            self.positions_direction[agent_id] = direction
        elif action == "TURN_RIGHT":
            direction = [direction[1], -direction[0]]
            self.positions_direction[agent_id] = direction
        elif action == "WAIT":
            pass
        else: # "FORWARD"
            dy, dx = direction  # 当前它的方向就是它的前进步长
            dy = dy  # 原点在左上角，且向下是x，向右是y
            dx = -dx #
            if self.obstacles[x + dx, y + dy] == self.config.FREE:
                if self.positions[x + dx, y + dy] == self.config.FREE:
                    self.positions[x, y] = self.config.FREE
                    x += dx
                    y += dy
                    self.positions[x, y] = self.config.OBSTACLE
            self.positions_xy[agent_id] = (x, y)

        # print(self.positions_xy[agent_id])
        # print(self.positions_direction[agent_id])

    def on_goal(self, agent_id):
        return self.positions_xy[agent_id] == self.finishes_xy[agent_id]

    def is_active(self, agent_id):
        return self.is_active[agent_id]

    def hide_agent(self, agent_id):
        if not self.is_active[agent_id]:
            return False
        self.is_active[agent_id] = False

        self.positions[self.positions_xy[agent_id]] = self.config.FREE

        return True

    def show_agent(self, agent_id):
        if self.is_active[agent_id]:
            return False

        self.is_active[agent_id] = True
        if self.positions[self.positions_xy[agent_id]] == self.config.OBSTACLE:
            raise KeyError("The cell is already occupied")
        self.positions[self.positions_xy[agent_id]] = self.config.OBSTACLE
        return True


class GridLifeLong(Grid):
    def __init__(self, grid_config: GridConfig, add_artificial_border: bool = True, num_retries=10):

        super().__init__(grid_config, add_artificial_border, num_retries)

        self.component_to_points, self.point_to_component = get_components(grid_config, self.obstacles,
                                                                           self.positions_xy, self.finishes_xy)

        for i in range(len(self.positions_xy)):
            position, target = self.positions_xy[i], self.finishes_xy[i]
            if self.point_to_component[position] != self.point_to_component[target]:
                warnings.warn(f"The start point ({position[0]}, {position[1]}) and the goal"
                              f" ({target[0]}, {target[1]}) are in different components. The goal is changed.",
                              Warning, stacklevel=2)
