from gymnasium import Wrapper


class AbstractMetric(Wrapper):
    def _compute_stats(self, step, is_on_goal, finished,  _conflict_nums):
        raise NotImplementedError

    def __init__(self, env):
        super().__init__(env)
        self._current_step = 0

    def step(self, action):
        obs, reward, terminated, truncated, infos = self.env.step(action)
        finished = all(truncated) or all(terminated)


        current_conflict_nums = [0 for _ in range(self.get_num_agents())]
        for agent_idx in range(self.get_num_agents()):
            current_conflict_nums[agent_idx] = infos[agent_idx]['current_conflict_nums']
        metric = self._compute_stats(self._current_step, self.was_on_goal, finished , current_conflict_nums)

        self._current_step += 1
        if finished:
            self._current_step = 0

        if metric:
            if 'metrics' not in infos[0]:
                infos[0]['metrics'] = {}
            infos[0]['metrics'].update(**metric)

        return obs, reward, terminated, truncated, infos


class LifeLongAverageThroughputMetric(AbstractMetric):

    def __init__(self, env):
        super().__init__(env)
        self._solved_instances = 0

    def _compute_stats(self, step, is_on_goal, finished, _conflict_nums):
        for agent_idx, on_goal in enumerate(is_on_goal):
            if on_goal:
                self._solved_instances += 1
        if finished:
            result = {'avg_throughput': self._solved_instances / self.grid_config.max_episode_steps}
            self._solved_instances = 0
            return result


class LifeLongAttritionMetric(AbstractMetric):

    def __init__(self, env):
        super().__init__(env)
        self._attrition_steps = 0
        self._on_goal_steps = 0

    def _compute_stats(self, step, is_on_goal, finished, _conflict_nums):
        for agent_idx, on_goal in enumerate(is_on_goal):
            if not on_goal:
                self._attrition_steps += 1
            else:
                self._on_goal_steps += 1
        if finished:
            result = {
                'attrition': self._attrition_steps / max(1, self._on_goal_steps)}
            self._solved_instances = 0
            return result


class NonDisappearCSRMetric(AbstractMetric):

    def _compute_stats(self, step, is_on_goal, finished, _conflict_nums):
        if finished:
            return {'CSR': float(all(is_on_goal))}


class NonDisappearISRMetric(AbstractMetric):

    def _compute_stats(self, step, is_on_goal, finished, _conflict_nums):
        if finished:
            return {'ISR': float(sum(is_on_goal)) / self.get_num_agents()}


class NonDisappearEpLengthMetric(AbstractMetric):

    def _compute_stats(self, step, is_on_goal, finished, _conflict_nums):
        if finished:
            return {'ep_length': step}


class EpLengthMetric(AbstractMetric):
    def __init__(self, env):
        super().__init__(env)
        self._solve_time = [None for _ in range(self.get_num_agents())]

    def _compute_stats(self, step, is_on_goal, finished, _conflict_nums):
        for idx, on_goal in enumerate(is_on_goal):
            if self._solve_time[idx] is None:
                if on_goal or finished:
                    self._solve_time[idx] = step

        if finished:
            result = {'ep_length': sum(self._solve_time) / self.get_num_agents() + 1}
            self._solve_time = [None for _ in range(self.get_num_agents())]
            return result


class CSRMetric(AbstractMetric):
    def __init__(self, env):
        super().__init__(env)
        self._solved_instances = 0

    def _compute_stats(self, step, is_on_goal, finished, _conflict_nums):
        self._solved_instances += sum(is_on_goal)
        if finished:
            results = {'CSR': float(self._solved_instances == self.get_num_agents())}
            self._solved_instances = 0
            return results


class ISRMetric(AbstractMetric):
    def __init__(self, env):
        super().__init__(env)
        self._solved_instances = 0

    def _compute_stats(self, step, is_on_goal, finished, _conflict_nums):
        self._solved_instances += sum(is_on_goal)
        if finished:
            results = {'ISR': self._solved_instances / self.get_num_agents()}
            self._solved_instances = 0
            return results


class ConflictNumsMetric(AbstractMetric):
    def __init__(self, env):
        super().__init__(env)
        self.total_conflict_nums = [0 for _ in range(self.get_num_agents())]

    def _compute_stats(self, step, is_on_goal, finished, _conflict_nums):
        self.total_conflict_nums = list(map(lambda x, y: x + y, self.total_conflict_nums, _conflict_nums))
        _solved_instances = sum(self.total_conflict_nums)
        if finished:
            results = {'conflict_nums': _solved_instances / self.get_num_agents()}
            self.total_conflict_nums = [0 for _ in range(self.get_num_agents())]
            return results