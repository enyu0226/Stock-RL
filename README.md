# Stock-RL

A simple POC stock trader program that is based on Actor Critic Reinforcement Learning algorithm.

## MDP formulation

In order to to model the stock trading problem as a RL problem, we would need to define it as as a MDP.
The formulated MDP is solved using Actor Critic.
The environment is the platform of exchanges, both humans and other trading machine alike.
The agent is our stock trading bot
The actions are buy, sell/short, do nothing signal.
The rewards are increased financial return, possibly as a result increased moving trend average.

## Why Actor Critic?

Actor Critic Algorithm is chosen over Deep Q Learning since it incorporates the advantages from both value and policy-based model, namely increasing stability of training and also faster convergence. The value of state function is calculated and approximated by the critic neural network and it is used as a baseline to subtract from the approximated value of state-action pair of a optimal Q-function that is computed by the actor neural network in order to obtain the advantage of each action, such a strategy is used to solve the problem of high variance that is introduced to the policy gradient during experience gathering training phase.

The model only gives buy, sell/short and do nothing signal. It does not take into starting account balance and only one stock can be purchase or sold for the duration of the episode.

## Usage

```
python trader.py --training training_data.csv --testing testing_data.csv --output output.csv
```

## Strategy

Use Actor Critic RL algorithm to predict next day's stock trend, and use simple judge policy to perform action according to the predicted trend.

### State of Actor Critic

- Moving average (10 days)
- Difference of today moving average & yesterday moving average
- Magnitude of difference of moving average: [0,1,2,3] (fall, decline, flat, rise)

### Predict Trend

- 0: fall
- 1: decline
- 2: flat
- 3: rise

### Hold State

- -1: short 1 stock
- 0: no stock
- 1: hold 1 stock

### Action

- 1: buy
- 0: no action
- -1: sold (or short)

### Action Strategy

```
if(hold == 0):
  if(trend_of_tomorrow > 1):
    action = 1
  else:
    action = -1
elif(hold == 1):
  if(trend_of_tomorrow > 1):
    action = 0
  else:
    action = -1
else:
  if(trend_of_tomorrow < 3):
    action = 0
  else:
    action = 1
```

---

## Stock Data Format

| Open       | High       | Low        | Close      |
| ---------- | ---------- | ---------- | ---------- |
| 209.894836 | 216.427353 | 207.758728 | 216.208771 |
| ...        | ...        | ...        | ...        |

## Action Type

The possible actions computed by the actors are as followed:
1 → Buy signal. If you have already short 1 unit of a given stock, you will return to 0 as the open price in the next day. If you did not have any unit, you will have 1 unit as the open price in the next day. If you already have 1 unit, the code will be terminated due to the invalid status.

0 → No action signal. If you have 1 unit of a given stock now, hold it. If your slot is available, the status continues. If you short 1 unit, the status continues.

-1 → Sell signal. If you own 1 unit of a given stock, your will return to 0 as the open price in the next day. If you did not have any unit, it will be shorted 1 unit as the open price in the next day. If you had already short 1 unit, the code will be terminated due to the invalid status.

On the final day, if you still hold a given stock, a forced sale/buyback would be performed based on the close price of the final day in the testing period and your slot would be empty. Finally, your account will be settled and your profit will be calculated.

## Output Sample:

1. Each line in output file contains the action type which will be executed in the opening of the next day.

2. If the testing data contains 300 lines, the output would include 299 lines. But the last day will be settled without executing the specified action, and the program will use the closing price of the last day as the settled price.
