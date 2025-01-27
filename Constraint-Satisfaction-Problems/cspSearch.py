import sys
sys.path.append('../SearchProblem')
from cspProblem import Constraint, CSP
from searchProblem import Arc, Search_problem
from cspDFS import dfs_solver



class Search_from_CSP(Search_problem):

    def __init__(self, csp, variable_order=None):

        self.csp = csp

        if(variable_order):
            assert set(variable_order) == set(csp.variables)
            assert len(variable_order) == len(csp.variables)
            self.variables = variable_order
        else:
            self.variables = list(csp.variables)

    def is_goal(self, node):

        return len(node) == len(self.csp.variables)

    def start_node(self):

        return self.variables[0]

    def neighbors(self, node):

        var = self.variables[len(node)]
        res = []

        for val in var.domain:
            new_env = node | {val: val}

            if(self.csp.consistent(new_end)):
                res.append(Arc(node, new_env))

        return res

    def search(self):

        return dfs_solver(self.csp.constraints, {}, self.variables)
        
def solver_from_searcher(csp):

    path = Searcher(Search_from_CSP(csp)).search() #generic searcher from Search_Problem

    if(path is not None):
        return path.end()
    else:
        return None
