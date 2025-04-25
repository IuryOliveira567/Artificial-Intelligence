import matplotlib.pyplot as plt
import random
import sys


sys.path.append("../utils")
from agent import Displayable

class Search_problem(Displayable):

      def start_node(self):

          raise NotImplementedError("start_node")

      def is_goal(self, node):

          raise NotImplementedError("is_goal")

      def neighbors(self, node):

          raise NotImplementedError("neighbors")

      def heuristic(self, n):

          return 0

class Arc(object):

    def __init__(self, from_node, to_node, cost=1, action=None):

        self.from_node = from_node
        self.to_node = to_node
        self.action = action
        self.cost = cost

        assert cost >= 0, (f"Cost cannot be negative: {self}, cost={cost}")

    def __repr__(self):

        if(self.action):
            return f"{self.from_node} -- {self.action}--> {self.to_node}"
        else:
            return f"{self.from_node} --> {self.to_node}"

class Search_problem_from_explicit_graph(Search_problem):

      def __init__(self, title, nodes, arcs, start=None, goals=set(),
                   hmap={}, positions=None, show_costs=True):

          self.title = title
          self.neighs = {}
          self.nodes = nodes
          self.arcs = arcs
          self.start = start
          self.goals = goals
          self.hmap = hmap
          
          for node in nodes:
              self.neighs[node] = []
          
          for arc in arcs:
              self.neighs[arc.from_node].append(arc)

          if positions is None:
              self.positions = {node: (random.random(), random.random()) for node in nodes}
          else:
              self.positions = positions

          self.show_costs = show_costs

      def start_node(self):

          return self.start

      def is_goal(self, node):

          return node in self.goals

      def neighbors(self, node):

          return self.neighs[node]

      def heuristic(self, node):

          if(node in self.hmap):
              return self.hmap[node]
          else:
              return 0

      def __repr__(self):

          res = ""
          
          for arc in self.arcs:
              res += f"{arc}."

          return res

      def show(self, fontsize=10, node_color="orange", show_costs=None):

          self.fontsize = fontsize

          if(show_costs is not None):

              self.show_costs = show_costs

          plt.ion()
          ax = plt.figure().gca()
          ax.set_axis_off()

          plt.title(self.title, fontsize=fontsize)
          self.show_graph(ax, node_color)

      def show_graph(self, ax, node_color="orange"):

          bbox = dict(boxstyle="round4,pad=1.0,rounding_size=0.5", facecolor=node_color)

          for arc in self.arcs:
              self.show_arc(ax, arc)

          for node in self.nodes:
              self.show_node(ax, node, node_color = node_color)

      def show_node(self, ax, node, node_color):

          x, y = self.positions[node]
          ax.text(x, y, node, bbox=dict(boxstyle="round4,pad=1.0,rounding_size=0.5",
                                        facecolor=node_color), ha='center', va='center',
                                        fontsize=self.fontsize)

      def show_arc(self, ax, arc, arc_color="black", node_color="white"):

          from_pos = self.positions[arc.from_node]
          to_pos = self.positions[arc.to_node]

          ax.annotate(arc.to_node, from_pos, xytext=to_pos,
          arrowprops={'arrowstyle':'<|-', 'linewidth': 2,
                      'color':arc_color},
          bbox=dict(boxstyle="round4,pad=1.0,rounding_size=0.5",
                    facecolor=node_color), ha='center',va='center', fontsize=self.fontsize)

          if(self.show_costs):
              ax.text((from_pos[0] + to_pos[0]) / 2, (from_pos[1] + to_pos[1]) / 2,
                      arc.cost, bbox=dict(pad=1, fc='w', ec='w'),
                      ha='center', va='center', fontsize=self.fontsize)
      
class Path(object):

    def __init__(self, initial, arc=None):

        self.initial = initial
        self.arc = arc

        if(arc is None):
            self.cost = 0
        else:
            self.cost = initial.cost + arc.cost

    def end(self):

        if(self.arc is None):
            return self.initial
        else:
            return self.arc.to_node

    def nodes(self):

        current = self
            
        while current.arc is not None:
            yield current.arc.to_node
            current = current.initial

        yield current.initial

    def initial_nodes(self):

        if(self.arc is not None):
            yield from self.initial_nodes()

    def __repr__(self):

        if(self.arc is None):
            return str(self.initial)
        elif(self.arc.action):
            return f"{self.initial}\n -- {self.arc.action}-->{self.arc.to_node}"
        else:
            return f"{self.initial} --> {self.arc.to_node}"
