from searchGeneric import AStarSearcher
from searchProblem import Path


class SearcherMPP(AStarSearcher):

    def __init__(self, problem):

        super().__init__(problem)
        self.explored = set()

    def search(self):

        while(not self.empty_frontier()):
            self.path = self.frontier.pop()

            if(self.path.end() not in self.explored):
                self.explored.add(self.path.end())
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

                    for arc in neighs:
                        self.add_to_frontier(Path(self.path, arc))

                    self.display(3, f"New frontier: {[p.end() for p in self.frontier]}")

        self.display(0, "No (more) solutions. Total of",
                     self.num_expanded, "paths expanded.")
