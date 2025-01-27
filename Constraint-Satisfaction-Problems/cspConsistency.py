import sys

sys.path.append("../utils")
from agent import Displayable

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
