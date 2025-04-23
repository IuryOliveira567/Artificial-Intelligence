import sys

from variable import Variable
from cspProblem import CSP, Constraint
from operator import lt
from cspDFS import dfs_solve1
from cspSearch import Search_from_CSP
from cspConsistency import Con_solver
from cspSLS import SLSearch, Runtime_distribution
from cspSoft import SoftConstraint, DF_branch_and_bound_opt
from cspConsistencyGUI import ConsistencyGUI


v1 = Variable("v1", {1, 2, 3})
v2 = Variable("v2", {3, 4, 5})
v3 = Variable("v3", {4, 5, 6})
v4 = Variable("v4", {4, 8, 0})

c1 = Constraint([v1, v2], lt)
c2 = Constraint([v2, v3], lt)
c3 = Constraint([v4, v1], lt)

#c1 = SoftConstraint([v1, v2], lt)
#c2 = SoftConstraint([v2, v3], lt)
#c3 = SoftConstraint([v4, v1], lt)


#v4 < v1 < v2 < v3

csp0 = CSP("csp0", [v1, v2, v3, v4], [c1, c2, c3])
#g = dfs_solve1(csp0)

#sfcsp = Search_from_CSP(csp0)
#ac_sp = Con_solver(csp0)
#f = ac_sp.generate_sols()
#sls = SLSearch(csp0)
#sls.search(10, 1)
#rtd = Runtime_distribution(csp0)
dbopt = DF_branch_and_bound_opt(csp0)
f = ConsistencyGUI(csp0)
