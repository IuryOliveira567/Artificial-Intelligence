from searchProblem import Path
from searchGeneric import Searcher

import sys

sys.path.append('../utils')
from agent import Displayable


class DF_branch_and_bound(Searcher):

    def __init__(self, problem, bound=float("inf")):

        super().__init__(problem)
        self.best_path = None
        self.bound = bound

    def search(self):

        self.frontier = [Path(self.problem.start_node())]
        self.num_expanded = 0

        while(self.frontier):
            self.path = self.frontier.pop()

            if(self.path.cost + self.problem.heuristic(self.path.end()) < self.bound):
                self.display(2, "Expanding: ", self.path, "cost:", self.path.cost)
                self.num_expanded += 1

                if(self.problem.is_goal(self.path.end())):
                    self.best_path = self.path
                    self.bound = self.path.cost
                    self.display(1, "New best path: ", self.path, "cost: ", self.path.cost)
                else:
                    neighs = self.problem.neighbors(self.path.end())
                    self.display(4, "Neighbors are", neighs)

                    for arc in reversed(list(neighs)):
                        self.add_to_frontier(Path(self.path, arc))

                    self.display(3, f"New frontier: {[p.end() for p in self.frontier]}")


        self.path = self.best_path
        self.solution = self.best_path
        self.display(1, f"Optional solution is {self.best_path}."
                     if self.best_path else "No solution found.",
                     f"Number of paths expanded: {self.num_expanded}.")

        return self.best_path
