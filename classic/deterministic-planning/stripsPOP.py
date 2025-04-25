import sys

sys.path.append('../SearchProblem')
sys.path.append('../utils')

from searchProblem import Arc, Search_problem
from agent import Displayable
import random


class Action_instance(object):

    next_index = 0

    def __init__(self, action, index=None):

        if(index is None):
            index = Action_instance.next_index
            Action_instance.next_index += 1

        self.action = action
        self.index = index

    def __str__(self):

        return f"{self.action}#{self.index}"

    __repr__ = __str__

class POP_node(object):

    def __init__(self, actions, constraints, agenda, causal_links):

        self.actions = actions
        self.constraints = constraints
        self.agenda = agenda
        self.causal_links = causal_links

    def __str__(self):

        return ("actions: " + str({str(a) for a in self.actions}) + "\nconstraints: " +
                str({(str(a1), str(a2)) for (a1, a2) in self.constraints}) + "\nagenda: " +
                str([(str(s), str(a)) for (s,a) in self.agenda]) + "\ncausal_links: " +
                str({(str(a0), str(g), str(a2)) for (a0, g, a2) in self.causal_links}))

    def extract_plan(self):

        sorted_acts = []
        other_acts = set(self.actions)

        while other_acts:
            a = random.choice([a for a in other_acts if
                               all(((a1, a) not in self.constraints) for a1 in
                                   other_acts)])

            sorted_acts.append(a)
            other_acts.remove(a)

        return sorted_acts

class POP_search_from_STRIPS(Search_problem, Displayable):

    def __init__(self, planning_problem):

        Search_problem.__init__(self)
        self.planning_problem = planning_problem
        self.start = Action_instance("start")
        self.finish = Action_instance("finish")

    def is_goal(self, node):

        return node.agenda == []

    def start_node(self):

        constraints = {(self.start, self.finish)}
        agenda = [(g, self.finish) for g in
                  self.planning_problem.goal.items()]

        return POP_node([self.start, self.finish], constraints, agenda, [])

    def neighbors(self, node):

        self.display(3, "finding neighbors of\n", node)

        if(node.agenda):
            subgoal, act1 = node.agenda[0]
            self.display(2, "selecting", subgoal, "for", act1)
            new_agenda = node.agenda[1:]

            for act0 in node.actions:
                if(self.achieves(act0, subgoal) and
                   self.possible((act0, act1), node.constraints)):
                    self.display(2, " reusing", act0)

                    consts1 = self.add_constraint((act0, act1), node.constraints)
                    new_clink = (act0, subgoal, act1)
                    new_cls = node.causal_links + [new_clink]

                    for consts2 in self.protect_cl_for_actions(node.actions, consts1, new_clink):
                            yield Arc(node,
                                      POP_node(node.actions, consts2, new_agenda, new_cls),
                                      cost=0)

            for a0 in self.planning_problem.prob_domain.actions:
                if(self.achieves(a0, subgoal)):
                    new_a = Action_instance(a0)
                    self.display(2, " using new action", new_a)
                    new_actions = node.actions + [new_a]

                    consts1 = self.add_constraint((self.start, new_a), node.constraints)
                    consts2 = self.add_constraint((new_a, act1), consts1)

                    new_agenda1 = new_agenda + [(pre, new_a) for pre in
                                                a0.preconds.items()]

                    new_clink = (new_a, subgoal, act1)
                    new_cls = node.causal_links + [new_clink]

                    for consts3 in self.protect_all_cls(node.causal_links, new_a, consts2):
                        for consts4 in self.protect_cl_for_actions(node.actions, consts3, new_clink):
                            yield Arc(node, POP_node(new_actions, consts4, new_agenda1, new_cls),
                                      cost=1)

    def protect_cl_for_actions(self, actions, constrs, clink):

        if(actions):
            a = actions[0]
            rem_actions = actions[1:]
            a0, subgoal, a1 = clink

            if(a != a0 and a != a1 and self.deletes(a, subgoal)):
                if(self.possible((a, a0), constrs)):
                    new_const = self.add_constraint((a, a0), constrs)
                    for e in self.protect_cl_for_actions(rem_actions, new_const, clink):
                        yield e

                if(self.possible((a1, a), constrs)):
                    new_const = self.add_constraint((a1, a), constrs)
                    for e in self.protect_cl_for_actions(rem_actions, new_const, clink):
                        yield e
            else:
                for e in self.protect_cl_for_actions(rem_actions, constrs, clink):
                    yield e
        else:
            yield constrs

    def protect_all_cls(self, clinks, act, constrs):

        if(clinks):
            (a0,cond, a1) = clinks[0]
            rem_clinks = clinks[1:]

            if(act != a0 and act != a1 and self.deletes(act, cond)):
                if(self.possible((act, a0), constrs)):
                    new_const = self.add_constraint((act, a0), new_const)

                    for e in self.protect_all_cls(rem_clinks, act, new_const):
                        yield e
                        
                if(self.possible((a1, act), constrs)):
                   new_const = self.add_constraint((a1, act), constrs)

                   for e in self.protect_all_cls(rem_clinks, act, new_const):
                       yield e
            else:
                for e in self.protect_all_cls(rem_clinks, act, constrs):
                    yield e
        else:
            yield constrs

    def achieves(self, action, subgoal):

        var, val = subgoal
        return var in self.effects(action) and self.effects(action)[var] == val

    def deletes(self, action, subgoal):

        var, val = subgoal
        return var in self.effects(action) and self.effects(action)[var] != val

    def effects(self, action):

        if(isinstance(action, Action_instance)):
            action = action.action

        if(action == "start"):
            return self.planning_problem.initial_state
        elif(action == "finish"):
            return {}
        else:
            return action.effects

    def add_constraint(self, pair, const):

        if(pair in const):
            return const

        to_do = [pair]
        new_const = const.copy()

        while(to_do):
            x0, x1 = to_do.pop()
            new_const.add((x0, x1))

            for x, y in new_const:
                if(x == x1 and (x0, y) not in new_const):
                    to_do.append((x0, y))

                if(y == x0 and (x, x1) not in new_const):
                    to_do.append((x, x1))

        return new_const

    def possible(self, pair, constraint):

        (x, y) = pair
        return (x, y) not in constraint
