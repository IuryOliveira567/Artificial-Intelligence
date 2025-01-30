from cspProblem import Variable, Constraint, CSP
from agent import Displayable
import math


class SoftConstraint(Constraint):

    def __init__(self, scope, function, string=None, position=None):

        Constraint.__init__(self, scope, function, string, position)

    def value(self, assignment):

        return self.holds(assignment)

class DF_branch_and_bound_opt(Displayable):

    def __init__(self, csp, bound=math.inf):

        super().__init__()
        self.csp = csp
        self.best_asst = None
        self.bound = bound

    def optimize(self):

        self.num_expanded = 0
        self.cbsearch({}, 0, self.csp.constraints)
        self.display(1, "Number of paths expanded : ", self.num_expanded)

        return self.best_asst, self.bound

    def cbsearch(self, asst, cost, constraints):

        self.display(2, "cbsearch : ", asst, cost, constraints)
        can_eval = [c for c in constraints if c.can_evaluate(asst)]
        rem_cons = [c for c in constraints if c not in can_eval]
        newcost = cost + sum(c.value(asst) for c in can_eval)

        self.display(2, "Evaluating : ", can_eval, "cost: ", newcost)

        if(newcost < self.bound):
            self.num_expanded += 1
            if(rem_cons == []):
                self.best_asst = asst
                self.bound = newcost
                self.display(1, "New best assignment : ", asst, " cost: ", newcost)
            else:
                var = next(var for var in self.csp.variables if var not in asst)

                for val in var.domain:
                    self.cbsearch({var: val} | asst, newcost, rem_cons)
