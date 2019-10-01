import numpy as np

MAX_MOVING_AVG_LEN = 10

# s_, r, done


class Env:
    def __init__(self, data):
        self.data = data
        self.moving_avg = []
        self.moving_avg_diff = self._cal_diff_range()
        self.mean_diff = self._cal_mean_diff(self.moving_avg_diff)
        self.mean_square_error = self._cal_mean_square_error(
            self.mean_diff, self.moving_avg_diff)

        self.trend_block = [
            (self.mean_square_error/4 - self.mean_diff),
            (-self.mean_square_error),
            (-self.mean_square_error/4)
        ]

    def get_trend_block(self):
        return self.trend_block

    def _cal_mean_diff(self, moving_avg_diff):
        data_len = len(moving_avg_diff)
        mean_diff = sum(moving_avg_diff) / data_len
        return mean_diff

    def _cal_mean_square_error(self, mean_diff, moving_avg_diff):
        data_len = len(moving_avg_diff)
        total_error = 0
        for day in range(data_len):
            total_error += (moving_avg_diff[day] - mean_diff)**2

        mean_square_error = (total_error / data_len)**0.5

        return mean_square_error

    def _cal_diff_range(self):
        moving_avg_diff = [0]
        for day in range(len(self.data['Open'])):
            moving_avg = self._cal_moving_avg(day)
            self.moving_avg.append(moving_avg)
            if (day > 0):
                moving_avg_diff.append(
                    self.moving_avg[day] - self.moving_avg[day - 1])
        return moving_avg_diff

    def _cal_moving_avg(self, day):
        total = 0
        for index in range(MAX_MOVING_AVG_LEN):
            if ((day - index) < 0):
                total += self.data['Open'][0]
            else:
                total += self.data['Open'][day-index]

        return (total / MAX_MOVING_AVG_LEN) - self.data['Open'][0]

    def _cal_avg_change_trend(self, _avg_diff):
        if (_avg_diff > self.trend_block[0]):
            _avg_change_period = 3
        elif(_avg_diff < self.trend_block[1]):
            _avg_change_period = 0
        elif(_avg_diff < self.trend_block[2]):
            _avg_change_period = 1
        else:
            _avg_change_period = 2
        return _avg_change_period

    def _get_new_state(self, day, state):
        _moving_avg = self._cal_moving_avg(day)
        _avg_diff = _moving_avg - state[0]
        _avg_change_trend = self._cal_avg_change_trend(_avg_diff)

        _state = [_moving_avg, _avg_diff, _avg_change_trend]
        return np.array(_state)

    def _get_new_reword(self, day, state, action, _state):
        if (action - _state[2] > 1 or action - _state[2] < -1):
            reward = -100
        elif (action == _state[2]):
            reward = 20
        else:
            reward = -10

        return reward

    def step(self, day, state, action):
        if(day == (len(self.data['Open'])) - 1):
            done = True
            _state = 0
            reword = 0
        else:
            done = False
            _state = self._get_new_state(day, state)
            reword = self._get_new_reword(day, state, action, _state)

        return (_state, reword, done)
