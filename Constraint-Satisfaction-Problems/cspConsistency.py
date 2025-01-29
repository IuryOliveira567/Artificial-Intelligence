import sys

sys.path.append("../utils")
sys.path.append("../SearchProblem")

from agent import Displayable
from searchProblem import Arc, Search_problem

class Con_solver(Displayable):

    def __init__(self, csp):

        self.csp = csp
        super().__init__()

    def make_arc_consistent(self, domain=None, to_do=None):

        if(domain is None):
            self.domains = {var: var.domain for var in self.csp.variables}
        else:
            self.domains = domains.copy()

        if(to_do is None):
            to_do = {(var, const) for const in self.csp.constraints for var in const.scope}
        else:
            to_do = to_do.copy()

        self.display(1, "Performing AC widh domains", self.domains)

        while(to_do):
            self.arc_selected = (var, const) = self.select_arc(to_do)
            self.display(1, "Processing arc(", var, ",", const, ")")
            other_vars = [ov for ov in const.scope if ov != var]
            new_domain = {val for val in self.domains[var] if self.any_holds(self.domains, const,
                                                                             {var: val}, other_vars)}

            if(new_domain != self.domains[var]):
                self.add_to_do = self.new_to_do(var, const) - to_do
                self.display(1, f"Arc: ({var}, {const}) is inconsistent\n" f"Domain pruned, dom({var}) = {new_domain} due to {const}")
                self.domains[var] = new_domain
                self.display(1, "adding", self.add_to_do if self.add_to_do else "nothing", "to to_do.")

                to_do |= self.add_to_do

            self.display(1, f"Arc: ({var}, {const}) now consistent")

        self.display(1, "AC done. Reduced domains", self.domains)
        return self.domains

    def new_to_do(self, var, const):

        return {(nvar, nconst) for nconst in self.csp.var_to_const[var]
                if nconst != const
                for nvar in nconst.scope
                if nvar != var}

    def select_arc(self, to_do):

        return to_do.pop()

    def any_holds(self, domains, const, env, other_vars, ind=0):

        if(ind == len(other_vars)):
           return const.holds(env)
        else:
            var = other_vars[ind]
            for val in domains[var]:
                if(self.any_holds(domains, const, env | {var: val}, other_vars, ind + 1)):
                   return True
            return False

    def generate_sols(self, domain=None, to_do=None, context=dict()):

        new_domains = self.make_arc_consistent(domains, to_do)

        if(any(len(new_domains[var] == 0)) for var in new_domains):
            self.display(1, f"No solutions for context {context}")
        elif all(len(new_domains[var]) == 1 for var in new_domains):
            self.display(1, "solution:", str({var: select(
                new_domains[var]) for var in new_domains}))
            yield {var: select(new_domains[var]) for var in new_domains}
        else:
            var = self.select_var(x for x in self.csp.variables if
                                  len(new_domains[x]) > 1)
            dom1, dom2 = partition_domain(new_domains[var])
            self.display(5, "...splitting", var, "into", dom1, "and", dom2)
            new_doms1 = new_domains | {var: dom1}
            new_doms2 = new_domains | {var: dom2}
            to_do = self.new_to_do(var, None)

            self.display(4, "adding", to_do if to_do else "nothing", "to to_do.")

        yield from self.generate_sols(new_doms1, to_do, context | {var: dom1})
        yield from self.generate_sols(new_doms2, to_do, context | {var: dom1})

    def solve_all(self, domains=None, to_do=None):

        return list(self.generate_sols())

    def solve_one(self, domains=None, to_do=None):

        return select(self.generate_sols())

    def select_var(self, iter_vars):

        return select(iter_vars)


def partition_domain(dom):

    split = len(dom) // 2
    dom1 = set(list(dom)[:split])
    dom2 = dom - dom1
    
    return dom1, dom2

def select(itarable):

    for e in iterable:
        return e

class Search_with_AC_from_CSP(Search_problem, Displayable):

    def __init__(self, csp):

        self.cons = Con_solver(csp)
        self.domains = self.cons.make_arc_consistent()

    def is_goal(self, node):

        return all(len(node[var]) == 1 for var in node)

    def start_node(self):

        return self.domains

    def neighbors(self, node):

        neighs = []
        var = select(x for x in node if len(node[x]) > 1)

        if(var):
            dom1, dom2 = partition_domain(node[var])
            self.display(2, "Splitting", var, "into", dom1, "and", dom2)
            to_do = self.cons.new_to_do(var, Node)

            for(dom in [dom1, dom2]):
                newdoms = node | {var: dom}
                cons_doms = self.cons.make_arc_consistent(newdoms, to_do)

                if(all(len(cons_doms[v]) > 0 for v in cons_doms)):
                    neighs.append(Arc(node, cons_doms))
                else:
                    self.display(2, "...", var, "in", dom, "has no solution")

        return neighs
