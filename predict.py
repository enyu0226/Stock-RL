import numpy as np

MAX_MOVING_AVG_LEN = 10


class predict:
    def __init__(self, trend_block):
        self.data = []
        self.trend_block = trend_block

    def push_data(self, price):
        self.data.append(price)

    def _cal_moving_avg(self, day):
        total = 0
        for index in range(MAX_MOVING_AVG_LEN):
            if ((day - index) < 0):
                total += self.data[0]
            else:
                total += self.data[day-index]

        return (total / MAX_MOVING_AVG_LEN) - self.data[0]

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

    def get_new_state(self, day, state):
        _moving_avg = self._cal_moving_avg(day)
        _avg_diff = _moving_avg - state[0]
        _avg_change_trend = self._cal_avg_change_trend(_avg_diff)

        _state = [_moving_avg, _avg_diff, _avg_change_trend]
        return np.array(_state)

    def action(self, hold, trend):
        if(hold == 0):
            if(trend > 1):
                action = 1
            else:
                action = -1
        elif(hold == 1):
            if(trend > 1):
                action = 0
            else:
                action = -1
        else:
            if(trend < 3):
                action = 0
            else:
                action = 1

        return action

    def check_money(self, hold, action, money, price):
        if (action == 1):
            money -= price
            if (hold == 0):
                hold = 1
            else:
                hold = 0

        elif (action == -1):
            money += price
            if (hold == 0):
                hold = -1
            else:
                hold = 0

        return (money, hold)
