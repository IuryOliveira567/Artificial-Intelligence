import random

class Variable(object):

    def __init__(self, name, domain, position=None):

        self.name = name
        self.domain = domain
        self.position = position if position else (random.random(), random.random())
        self.size = len(domain)

    def __str__(self):

        return self.name

    def __repr__(self):

        return self.name
