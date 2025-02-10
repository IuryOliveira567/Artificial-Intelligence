class Strips(object):

    def __init__(self, name, preconds, effects, cost=1):

        self.name = name
        self.preconds = preconds
        self.effects = effects
        self.cost = cost

    def __repr__(self):

        return self.name
    
    
class STRIPS_domain(object):

    def __init__(self, feature_domain_dict, actions):

        self.feature_domain_dict = feature_domain_dict
        self.actions = actions

class Planning_problem(object):

    def __init__(self, prob_domain, initial_state, goal):

        self.prob_domain = prob_domain
        self.initial_state = initial_state
        self.goal = goal

