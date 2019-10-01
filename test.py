import pandas as pd
import numpy as np
import tensorflow as tf

from env import Env
from predict import predict

from rl import (Actor, Critic)

LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning rate for critic
MAX_EPISODE = 25
N_F = 3  # mean & curve & block
N_A = 5  # 0, 1, 2, 3, 4

Train_df = pd.read_csv('./data/google/train.csv',
                       names=["Open", "High", "Low", "Close"])
Test_df = pd.read_csv('./data/google/test.csv',
                      names=["Open", "High", "Low", "Close"])
env = Env(Train_df)

sess = tf.Session()

actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
# we need a good teacher, so the teacher should learn faster than the actor
critic = Critic(sess, n_features=N_F, lr=LR_C)

sess.run(tf.global_variables_initializer())

# Training
for i_episode in range(MAX_EPISODE):
    s = np.array([0, 0, 0])
    t = 0
    track_r = []
    while True:
        a = actor.choose_action(s)

        s_, r, done = env.step(t, s, a)

        if (done):
            ep_rs_sum = sum(track_r)
            print("episode:", i_episode, "  reward:", int(ep_rs_sum))
            break

        else:
            track_r.append(r)

            # gradient = grad[r + gamma * V(s_) - V(s)]
            td_error = critic.learn(s, r, s_)
            # true_gradient = grad[logPi(s,a) * td_error]
            actor.learn(s, a, td_error)

            s = s_
            t += 1


s = np.array([0, 0, 0])

error = 0
correct = 0
error_1 = 0
error_2 = 0

# user state
trend_block = env.get_trend_block()
predict = predict(trend_block)
hold = 0
money = 0

for day in range(len(Test_df['Open'])):
    # Predict Trend
    trend = actor.choose_action(s)

    # Buy or Sold
    action = predict.action(hold, trend)
    #
    # New Day
    #
    price = Test_df['Open'][day]
    predict.push_data(price)

    # Check money after day 0 (day start from 0)
    if(day > 0):
        print("day: ", day - 1, "state: ", s, " ------ today: ", (s[2] - 2), "predict tomorrow: ", (
            trend - 2), " ------- hold: ", hold, " action tomorrow: ", action, "money: ", money)
        money, hold = predict.check_money(hold, action, money, price)

    s = predict.get_new_state(day, s)

    if(day + 1 == len(Test_df['Open'])):
        print("day: ", day, "state: ", s, " ------ today: ", (s[2] - 2), "predict tomorrow: ", (
            trend - 2), " ------- hold: ", hold, " action tomorrow: ", action, "money: ", money)
        if (hold == 1):
            money += Test_df['Close'][day]
        elif (hold == -1):
            money -= Test_df['Close'][day]
        print("final money: ", money)

    # error check
    if(np.absolute(trend - s[2]) == 0):
        correct += 1
    elif(np.absolute(trend - s[2]) == 1):
        print("1 -- predict: ", trend - 2, " real: ", s[2] - 2)
        error_1 += 1
    else:
        print("2 -- predict: ", trend - 2, " real: ", s[2] - 2)
        error_2 += 1

print("correct: ", correct)
print("error_1: ", error_1)
print("error_2: ", error_2)
