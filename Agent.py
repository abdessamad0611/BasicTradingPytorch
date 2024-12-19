import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
from collections import deque
import sys
import os

# Define the StockModel using PyTorch
class StockModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(StockModel, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 8)
        self.fc4 = nn.Linear(8, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)  # Linear activation for Q-values
        return x

# Define the Agent class
class Agent:

    rewards = []

    def __init__(self, state_size, is_eval=False, model_name=""):
        self.state_size = state_size  # normalized previous days
        self.action_size = 3  # sit, buy, sell
        self.memory = deque(maxlen=1000)
        self.inventory = []
        self.model_name = model_name
        self.is_eval = is_eval

        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Decay rate for exploration

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = torch.load("models/" + model_name) if is_eval else self._model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def _model(self):
        model = StockModel(self.state_size, self.action_size).to(self.device)
        return model

    def act(self, state):
        state = torch.FloatTensor(state).to(self.device)
        if not self.is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def stockRewards(self, reward):
      
        self.rewards.append(reward)

    def expReplay(self, batch_size):
      mini_batch = random.sample(self.memory, min(len(self.memory), batch_size))

      for state, action, reward, next_state, done in mini_batch:
          target = reward
          if not done:
              with torch.no_grad():
                  target = reward + self.gamma * torch.max(self.model(next_state)).item()

          # Get predictions for the current state
          target_f = self.model(state)

          # Ensure target_f is detached for indexing
          target_f = target_f.clone().detach()

          # Update the specific action with the target value
          target_f[0, action] = target

          # Perform optimization step
          self.optimizer.zero_grad()
          loss = self.criterion(self.model(state), target_f.unsqueeze(0))
          loss.backward()
          self.optimizer.step()

      # Decrease epsilon for exploration-exploitation tradeoff
      if self.epsilon > self.epsilon_min:
          self.epsilon *= self.epsilon_decay


    def getRewards(self):
        rewards = [reward for _, _, reward, _, _ in self.memory if reward > 0]
        return rewards

    def getAgentsrewards(self):
        return self.rewards

# Utility functions
def formatPrice(n):
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

def getStockDataVec(key):
    vec = []
    with open(f"./data/{key}.csv", "r") as file:
        lines = file.read().splitlines()

    for line in lines[1:]:
        vec.append(float(line.split(",")[4]))

    return vec

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def getState(data, t, n):
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1]  # Pad with t0
    res = []
    for i in range(n - 1):
        res.append(sigmoid(block[i + 1] - block[i]))

    return np.array([res])

def loadData(stockname):
    data = getStockDataVec(stockname)
    print(len(data))
    state = getState(data, 0, 4)
    t = 0
    d = t - 4

    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1]
    print('------------ Minus')
    print(-d * [data[0]] + data[0:t + 1])
    print('------------ State')
    print(state)
    print('------------  Block')
    res = []
    for i in range(3):
        res.append(sigmoid(block[i + 1] - block[i]))
    print(block)
    return 0



# Ensure the "models" directory exists
os.makedirs("models", exist_ok=True)


total_profitl = []
buy_info = []
sell_info = []
data_Store = []

stock_name, window_size, episode_count = 'GOLD', 3, 10

agent = Agent(window_size)  # Use the updated PyTorch Agent class
data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 32

for e in range(episode_count + 1):
    print("Episode " + str(e) + "/" + str(episode_count))
    state = torch.tensor(getState(data, 0, window_size + 1), dtype=torch.float32)

    total_profit = 0
    agent.inventory = []

    for t in range(l):
        # Sample a random action in the first episodes
        # and then predict the best action for a given state
        action = agent.act(state)

        # sit
        next_state = torch.tensor(getState(data, t + 1, window_size + 1), dtype=torch.float32)
        reward = 0

        if action == 1:  # buy
            agent.inventory.append(data[t])
            print("Buy: " + formatPrice(data[t]))

            # Save results for visualization
            buy_info.append(data[t])
            d = str(data[t]) + ', ' + 'Buy'
            data_Store.append(d)

        elif action == 2 and len(agent.inventory) > 0:  # sell
            bought_price = agent.inventory.pop(0)
            reward = max(data[t] - bought_price, 0)
            total_profit += data[t] - bought_price

            print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))
            total_profitl.append(data[t] - bought_price)

            step_price = data[t] - bought_price

            info = str(data[t]) + ',' + str(step_price) + ',' + str(reward)
            sell_info.append(info)
            d = str(data[t]) + ', ' + 'Sell'
            data_Store.append(d)

        done = True if t == l - 1 else False
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            print("--------------------------------")
            print("Total Profit: " + formatPrice(total_profit))
            print("--------------------------------")

        if len(agent.memory) > batch_size:
            agent.expReplay(batch_size)

    # Save the PyTorch model every 10 episodes
    if e % 10 == 0:
        torch.save(agent.model.state_dict(), f"models/model_ep{e}.pth")
