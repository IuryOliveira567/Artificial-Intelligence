import matplotlib.pyplot as plt

class plot_history(object):

    def __init__(self, ag, env):

        self.ag = ag
        self.env = env
        plt.ion()
        plt.xlabel("Time")
        plt.ylabel("Value")

    def plot_env_hist(self):

        num = len(self.env.stock_history)
        plt.plot(range(num), self.env.price_history, label="Price")
        plt.plot(range(num), self.env.stock_history, label="In stock")
        plt.legend()
        plt.draw()

    def plot_agent_hist(self):

        num = len(self.ag.buy_history)
        plt.bar(range(1, num + 1), self.ag.buy_history, label="Bought")
        plt.legend()
        plt.draw()
