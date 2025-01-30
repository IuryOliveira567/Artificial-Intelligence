from cspProblem import CSP, Constraint
import sys

sys.path.append("../SearchProblem")
sys.path.append("../utils")

from searchProblem import Arc, Search_problem
from agent import Displayable
import random
import heapq
import matplotlib.pyplot as plt


class SLSearch(Displayable):

    def __init__(self, csp):

        self.csp = csp
        self.variables_to_select = {var for var in self.csp.variables
                                    if len(var.domain) > 1}

        self.current_assignment = None
        self.number_of_steps = 0

    def restart(self):

        self.current_assignment = {var: random_choice(var.domain) for
                                   var in self.csp.variables}
        self.display(2, "Initial assignment", self.current_assignment)
        self.conflicts = set()

        for con in self.csp.constraints:
            if(not con.holds(self.current_assignment)):
                self.conflicts.add(con)

        self.display(2, "Number of conflicts", len(self.conflicts))
        self.variable_pq = None

    def search(self, max_steps, prob_best=0, prob_anycon=1.0):

        if(self.current_assignment is None):
            self.restart()
            self.number_of_steps += 1

            if(not self.conflicts):
                self.display(1, "Solution found : ", self.current_assignment,
                             "after restart")

                return self.number_of_steps

        if(prob_best > 0):
            return self.search_with_var_pq(max_steps, prob_best,
                                           prob_anycon)
        else:
            return self.search_with_any_conflict(max_steps, prob_anycon)

    def search_with_any_conflict(self, max_steps, prob_anycon=1.0):

        self.variable_pq = None

        for i in range(max_steps):
            self.number_of_steps += 1

            if(random.random() < prob_anycon):
                con = random_choice(self.conflicts)
                var = random_choice(con.scope)
            else:
                var = random_choice(self.variables_to_select)

            if(len(var.domain) > 1):
                val = random_choice([val for val in var.domain
                                     if val is not self.current_assignment[var]])
                self.display(2, self.number_of_steps, ": Assignment", var, "=", val)
                self.current_assignment[var] = val

                for varcon in self.csp.var_to_const[var]:
                    if(varcon.holds(self.current_assignment)):
                       if(varcon in self.conflicts):
                         self.conflicts.remove(varcon)
                    else:
                        if(varcon not in self.conflicts):
                            self.conflicts.add(varcon)

                self.display(2, " Number of conflicts", len(self.conflicts))

            if(not self.conflicts):
                self.display(1, "Solution found : ", self.current_assignment,
                             "in", self.number_of_steps, "steps")

                return self.number_of_steps

        self.display(1, "No solution in", self.number_of_steps, "steps",
                     len(self.conflicts), "conflicts remain")
        
        return None

    def search_with_var_pq(self, max_steps, prob_best=1.0, prob_anycon=1.0):

        if(not self.variable_pq):
            self.create_pq()

        pick_best_or_con = prob_best + prob_anycon

        for i in range(max_steps):
            self.number_of_steps += 1
            randnum = random.random()

            if(randnum < prob_best):
                var, oldval = self.variable_pq.top()
            elif(randnum < pick_best_or_con):
                con = random_choice(self.conflicts)
                var = random_choice(con.scope)
            else:
                var = random_choice(self.variables_to_select)
            if(len(var.domain) > 1):
                val = random_choice([val for val in var.domain if val is not
                                     self.current_assignment[var]])
                
                self.display(2, "Assignment", var, val)
                var_differential = {}
                self.current_assignment[var] = val

                for varcon in self.csp.var_to_const[var]:
                    self.display(3, "Checking", varcon)
                    if(varcon.holds(self.current_assignment)):
                       if(varcon in self.conflicts):
                          self.display(3, "Became consistent", varcon)
                          self.conflicts.remove(varcon)

                          for v in varcon.scope:
                            var_differential[var] = var_differential.get(v, 0) - 1
                    else:
                        if(varcon not in self.conflicts):
                            self.display(3, "Became inconsistent", varcon)
                            self.conflicts.add(varcon)

                            for v in varcon.scope:
                                var_differential[var] = var_differential.get(v, 0) + 1
                
                self.variable_pq.update_each_priority(var_differential)
                self.display(2, "Number of conflicts", len(self.conflicts))

            if(not self.conflicts):
                self.display(1, "Solution found : ", self.current_assignment, "in",
                             self.number_of_steps, "steps")

                return self.number_of_steps

        self.display(1, "No solution in", self.number_of_steps, "steps",
                         len(self.conflicts), "conflicts remain")
        return None

    def create_pq(self):    

        self.variable_pq = Updatable_priority_queue()
        var_to_number_conflicts = {}

        for con in self.conflicts:
            for var in con.scope:
                var_to_number_conflicts[var] = var_to_number_conflicts.get(var, 0) + 1

        for var, num in var_to_number_conflicts.items():
            print(var, num)
            if(num > 0):
                self.variable_pq.add(var, -num)

def random_choice(st):

    return random.choice(tuple(st))

class Updatable_priority_queue(object):

    def __init__(self):

        self.pq = []
        self.elt_map = {}
        self.REMOVED = "*removed*"
        self.max_size = 0

    def add(self, elt, val):

        assert val <= 0, val
        assert elt not in self.elt_map, elt

        new_triple = [val, random.random(), elt]
        heapq.heappush(self.pq, new_triple)
        self.elt_map[elt] = new_triple

    def remove(self, elt):

        if(elt in self.elt_map):
            self.elt_map[elt][2] = self.REMOVED
            del self.elt_map[elt]

    def update_each_priority(self, update_dict):

        for elt, incr in update_dict.items():
            if(incr != 0):
                newval = self.elt_map.get(elt, [0])[0] - incr
                assert newval <= 0, f"{elt}:{newval + incr} - {incr}"
                self.remove(elt)

                if(newval != 0):
                    self.add(elt, newval)

    def pop(self):

        self.max_size = max(self.max_size, len(self.pq))
        triple = heapq.heappop(self.pq)

        while(triple[2] == self.REMOVED):
            triple = heapq.heappop(self.pq)

        del self.elt_map[triple[2]]

        return triple[2], triple[0]

    def top(self):

        self.max_size = max(self.max_size, len(self.pq))
        triple = self.pq[0]

        while(triple[2] == self.REMOVED):
            heapq.heappop(self.pq)
            triple = self.pq[0]

        return triple[2], triple[0]

    def empty(self):

        return all(triple[2] == self.REMOVED for triple in self.pq)

class Runtime_distribution(object):

    def __init__(self, csp, xscale='log'):

        self.csp = csp
        plt.ion()
        plt.xlabel("Number of Steps")
        plt.ylabel("Cumulative Number of Runs")
        plt.xscale(xcale)

    def plot_runs(self, num_runs=100, max_steps=1000, prob_best=1.0, prob_anycon=1.0):

        stats = []
        SLSearcher.max_display_level, temp_mdl = 0, SLSearcher.max_display_level

        for i in range(num_runs):
            searcher = SLSearcher(self.csp)
            num_steps = searcher.search(max_steps, prob_best, prob_anycon)

            if(num_steps):
                stats.append(num_steps)

        stats.sort()

        if(prob_best > 1.0):
            label = "P(best)=1.0"
        else:
            p_ac = min(prob_anycon, 1 - prob_best)
            label = "P(best)=%.2f, P(ac)=%.2f" %(prob_best, p_ac)

        plt.plot(stats, range(len(stats)), label=label)
        plt.legend(loc="upper left")
        SLSearcher.max_display_level = temp_mdl
