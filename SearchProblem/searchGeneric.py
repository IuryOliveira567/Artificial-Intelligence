import heapq
from searchProblem import Path

import sys
sys.path.append('../utils')
from agent import Displayable


class Searcher(Displayable):

    def __init__(self, problem):

        self.problem = problem
        self.initialize_frontier()
        self.num_expanded = 0

        self.add_to_frontier(Path(problem.start_node()))
        super().__init__()

    def initialize_frontier(self):

        self.frontier = []

    def empty_frontier(self):

        return self.frontier == []

    def add_to_frontier(self, path):

        self.frontier.append(path)

    def search(self):

        while(not self.empty_frontier()):
            self.path = self.frontier.pop()
            self.num_expanded += 1

            if(self.problem.is_goal(self.path.end())):
                self.solution = self.path
                self.display(1, f"Solution: {self.path} (cost: {self.path.cost})\n",
                             self.num_expanded, "paths have been expanded and",
                             len(self.frontier), "paths remain in the frontier")

                return self.path
            else:
                self.display(4, f"Expanding: {self.path} (cost: {self.path.cost})")
                neighs = self.problem.neighbors(self.path.end())
                self.display(2, f"Expanding: {self.path} with neighbors {neighs}")

                for arc in reversed(list(neighs)):
                    self.add_to_frontier(Path(self.path, arc))

                self.display(3, f"New frontier: {[p.end() for p in self.frontier]}")

        self.display(0, "No (more) solutions. Total of",
                     self.num_expanded, "paths expanded.")

class FrontierPQ(object):

    def __init__(self):

        self.frontier_index = 0
        self.frontierpq = []

    def empty(self):

        return self.frontierpq == []

    def add(self, path, value):

        self.frontier_index += 1
        heapq.heappush(self.frontierpq, (value, -self.frontier_index, path))

    def pop(self):
        self.frontierpq
        (_, _, path) = heapq.heappop(self.frontierpq)
        return path

    def count(self, val):

        return sum(1 for e in self.frontierpq if e[0] == val)

    def __repr__(self):

        return str([(n, c, str(p)) for (n, c, p) in self.frontierpq])

    def __len__(self):

        return len(self.frontierpq)

    def __iter__(self):

        for(_, _, path) in self.frontierpq:
            yield path

class AStarSearcher(Searcher):

    def __init__(self, problem):

        super().__init__(problem)

    def initialize_frontier(self):

        self.frontier = FrontierPQ()

    def empty_frontier(self):

        return self.frontier.empty()

    def add_to_frontier(self, path):

        value = path.cost + self.problem.heuristic(path.end())
        self.frontier.add(path, value)
