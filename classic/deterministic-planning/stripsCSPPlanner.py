import sys

sys.path.append('../Constraint-Satisfaction-Problems')
from cspProblem import Variable, CSP, Constraint
from cspConsistency import Con_solver


class CSP_from_STRIPS(CSP):

    def __init__(self, planning_problem, number_stages=2):

        prob_domain = planning_problem.prob_domain
        initial_state = planning_problem.initial_state
        goal = planning_problem.goal

        self.action_vars = [Variable(f"Action{t}", prob_domain.actions)
                            for t in range(number_stages)]

        feat_time_var = {feat: [Variable(f"{feat}_{t}", dom)
                                for t in range(number_stages + 1)]
                         for(feat, dom) in prob_domain.feature_domain_dict.items()}

        constraints = [Constraint((feat_time_var[feat][0],), is_(val))
                       for(feat, val) in initial_state.items()]

        constraints += [Constraint((feat_time_var[feat][number_stages],),
                                    is_(val))
                        for(feat, val) in goal.items()]

        constraints += [Constraint((feat_time_var[feat][t],
                                     self.action_vars[t]),
                                     if_(val, act))
                                    for act in prob_domain.actions
                                    for(feat, val) in act.preconds.items()
                                    for t in range(number_stages)]

        constraints += [Constraint((feat_time_var[feat][t + 1],
                                    self.action_vars[t]),
                                   if_(val, act))
                        for act in prob_domain.actions
                        for feat, val in act.effects.items()
                        for t in range(number_stages)]

        constraints += [Constraint((feat_time_var[feat][t],
                                    self.action_vars[t], feat_time_var[feat][t + 1]),
                                   eq_if_not_in_({act for act in
                                                 prob_domain.actions
                                                 if(feat in act.effects)}))
                                   for feat in prob_domain.feature_domain_dict
                                   for t in range(number_stages)]

        variables = set(self.action_vars) | {feat_time_var[feat][t]
                                            for feat in prob_domain.feature_domain_dict
                                            for t in range(number_stages + 1)}

        CSP.__init__(self, "CSP_from_Strips", variables, constraints)

    def extract_plan(self, soln):

        return [soln[a] for a in self.action_vars]

def is_(val):

    def is_fun(x):
        
        return x == val

    is_fun.__name__ = f"value_is{val}"
    return is_fun

def if_(v1, v2):

    def if_fun(x1, x2):
        return x1 == v1 if x2 == v2 else True

    if_fun.__name__ = f"if x2 is {v2} then x1 is {v1}"
    return if_fun

def eq_if_not_in_(actset):

    def eq_if_not_fun(x1, a, x2):
        return x1 == x2 if a not in actset else True

    eq_if_not_fun.__name__ = f"first and third arguments are equal if action is not in {actset}"
    return eq_if_not_fun

def con_plan(prob, horizon):

    csp = CSP_from_STRIPS(prob, horizon)
    sol = Con_solver(csp).solve_one()

    return csp.extract_plan(sol) if sol else sol                    
