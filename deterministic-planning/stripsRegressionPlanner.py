import sys

sys.path.append('../SearchProblem')
from searchProblem import Arc, Search_problem
from stripsForwardPlanner import zero


class Subgoal(object):

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

class Regression_STRIPS(Search_problem):

    def __init__(self, planning_problem, heur=zero):

        self.prob_domain = planning_problem.prob_domain
        self.top_goal = Subgoal(planning_problem.goal)
        self.initial_state = planning_problem.initial_state
        self.heur = heur

    def is_goal(self, subgoal):

        goal_asst = subgoal.assignment
        return all(self.initial_state[g] == goal_asst[g]
                   for g in goal_asst)

    def start_node(self):

        return self.top_goal

    def neighbors(self, subgoal):

        goal_asst = subgoal.assignment
        return [Arc(subgoal, self.weakest_precond(act, goal_asst),
                    act.cost, act)
                for act in self.prob_domain.actions
                if self.possible(act, goal_asst)]

    def possible(self, act, goal_asst):

        return(any(goal_asst[prop] == act.effects[prop]
                   for prop in act.effects if prop in goal_asst)
               and all(goal_asst[prop] == act.effects[prop]
                       for prop in act.effects if prop in goal_asst)
               and all(goal_asst[prop] == act.preconds[prop]
                       for prop in act.preconds if prop not in act.effects
                       and prop in goal_asst))

    def weakest_precond(self, act, goal_asst):

        new_asst = act.preconds.copy()

        for g in goal_asst:
            if(g not in act.effects):
                new_asst[g] = goal_asst[g]

        return Subgoal(new_asst)

    def heuristic(self, subgoal):

        return self.heur(self.initial_state, subgoal.assignment)
