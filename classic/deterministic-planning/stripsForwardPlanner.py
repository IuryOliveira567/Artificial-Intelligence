import sys
from stripsProblem import Strips, STRIPS_domain

sys.path.append('../SearchProblem')
from searchProblem import Arc, Search_problem


class State(object):

    def __init__(self, assignment):

        self.assignment = assignment
        self.hash_value = None

    def __hash__(self):

        if(self.hash_value is None):
            self.hash_value = hash(frozenset(self.assignment.items()))

        return self.hash_value

    def __eq__(self, st):

        return self.assignment == st.assignment

    def __str__(self):

        return str(self.assignment)

def zero(*args, **nargs):

    return 0

class Forward_STRIPS(Search_problem):

    def __init__(self, planning_problem, heur=zero):

        self.prob_domain = planning_problem.prob_domain
        self.initial_state = State(planning_problem.initial_state)
        self.goal = planning_problem.goal
        self.heur = heur

    def is_goal(self, state):

        return all(state.assignment[prop] == self.goal[prop]
                   for prop in self.goal)

    def start_node(self):

        return self.initial_state

    def neighbors(self, state):

        return [Arc(state, self.effect(act, state.assignment), act.cost,
                    act)
                for act in self.prob_domain.actions
                if(self.possible(act, state.assignment))]

    def possible(self, act, state_asst):

        return all(state_asst[pre] == act.preconds[pre]
                   for pre in act.preconds)

    def effect(self, act, state_asst):

        new_state_asst = state_asst.copy()
        new_state_asst.update(act.effects)

        return State(new_state_asst)

    def heuristic(self, state):

        return self.heur(state.assignment, self.goal)
