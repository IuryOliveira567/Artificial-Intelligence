import sys

sys.path.append("../utils")
from agent import Displayable


class Clause(object):

    def __init__(self, head, body=[]):

        self.head = head
        self.body = body

    def __repr__(self):

        if(self.body):
            return f"{self.head} <- {' & '.join(str(a) for a in self.body)}."
        else:
            return f"{self.head}"

class Askable(object):

    def __init__(self, atom):

        self.atom = atom
    
    def __str__(self):

        return "askable " + self.atom + "."

def yes(ans):

    return ans.lower() in ['yes', 'oui', 'y']

class KB(Displayable):

    def __init__(self, statements=[]):

        self.statements = statements
        self.clauses = [c for c in statements if isinstance(c, Clause)]
        self.askables = [c.atom for c in statements if isinstance(c, Askable)]

        self.atom_to_clauses = {}

        for c in self.clauses:
            self.add_clause(c)
    
    def add_clause(self, c):

        if(c.head in self.atom_to_clauses):
            self.atom_to_clauses[c.head].append(c)
        else:
            self.atom_to_clauses[c.head] = [c]

    def clauses_for_atom(self, a):

        if(a in self.atom_to_clauses):
            return self.atom_to_clauses[a]
        else:
            return []

    def __str__(self):

        return '\n'.join([str(c) for c in self.statements])
