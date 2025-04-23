class Displayable(object):

  max_display_level = 1
  
  def display(self, level, *args, **nargs):
      if level <= self.max_display_level:
          print(*args, **nargs)

  
class Agent(Displayable):
  
    def initial_action(self, percept):
      
        return self.select_action(percept)

    def select_action(self, percept):
      
        raise NotImplementedError("go")


class Environment(Displayable):
  
    def initial_percept(self):
      
        raise NotImplementedError("initial_percept")

    def do(self, action):
      
        raise NotImplementedError("Environment.do")

class Simulate(Displayable):
  
    def __init__(self, agent, environment):
      
        self.agent = agent
        self.env = environment
        self.percept = self.env.initial_percept()
        self.percept_history = [self.percept]
        self.action_history = []

    def go(self, n):
      
        for i in range(n):
            action = self.agent.select_action(self.percept)
            self.display(2, f"i={i} action={action}")
            self.percept = self.env.do(action)
            self.display(2, f"   percept={self.percept}")

