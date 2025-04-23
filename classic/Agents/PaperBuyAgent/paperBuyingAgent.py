import random
import matplotlib.pyplot as plt
import sys

sys.path.append('../../utils')
from agent import Environment, Agent, Simulate
from utilities import select_from_dist
from PlotHistory import plot_history 

class TP_env(Environment):
    
    price_delta = [0, 0, 0, 21, 0, 20, 0, -64, 0, 0, 23, 0, 0, 0, -35, 17,
                   0, 76, 0, -41, 0, 0, 0, 21, 0, 5, 0, 5, 0, 0, 0, 5, 0,
                   -15, 0, 5, 18, 0, 5, 0, -115, 0, 115, 0, 5, 0, -15, 0, 5,
                   0, 5, 0, 0, 0, 5, 0, 19 -59, 0, 44, 0, 5, 0, 5, 0, 0, 0,
                   5, 0, -65, 50, 0, 5, 0, 5, 0, 0, 20, 0, 5, 0
                   ]

    def __init__(self, sd=10):

        self.sd = sd
        self.time = 0
        self.stock = 20
        self.stock_history = []
        self.price_history = []

    def initial_percept(self):
        
        self.stock_history.append(self.stock)
        self.price = round(234 + self.sd * random.gauss(0, 1))
        self.price_history.append(self.price)

        return {'price': self.price,
                'instock': self.stock
               }

    def do(self, action):

        used = select_from_dist({7:0.1, 6:0.2, 5:0.2, 4:0.3, 3:0.1, 2:0.1})
        bought = action['buy']
        self.stock = self.stock + bought - used
        self.stock_history.append(self.stock)
        self.time += 1
        self.price = round(self.price + self.price_delta[self.time % len(self.price_delta)] + self.sd * random.gauss(0,1))
        self.price_history.append(self.price)

        return {'price': self.price, 'instock': self.stock}

class TP_agent(Agent):

    def __init__(self):

        self.spent = 0
        percept = env.initial_percept()
        self.ave = self.last_price = percept['price']
        self.instock = percept['instock']
        self.buy_history = []

    def select_action(self, percept):

        self.last_price = percept['price']
        self.ave = self.ave + (self.last_price - self.ave) * 0.05
        self.instock = percept['instock']

        if(self.last_price < 0.9 * self.ave and self.instock < 60):
          tobuy = 48
        elif(self.instock < 12):
          tobuy = 12
        else:
          tobuy = 0

        self.spent += tobuy * self.last_price
        self.buy_history.append(tobuy)

        return {'buy': tobuy}

if __name__ == "__main__":
  env = TP_env()
  ag = TP_agent()
  sim = Simulate(ag, env)
  sim.go(90)
  
  pl = plot_history(ag,env)
  pl.plot_env_hist()
  pl.plot_agent_hist()
