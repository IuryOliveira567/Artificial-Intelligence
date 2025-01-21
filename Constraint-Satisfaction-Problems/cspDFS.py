import cspExamples
import time

def dfs_solver(constraints, context, var_order):

    to_eval = {c for c in constraints if c.can_evaluate(context)}
    
    if(all(c.holds(context) for c in to_eval)):
       if(var_order == []):
         yield context
       else:
         rem_cons = [c for c in constraints if c not in to_eval]
         var = var_order[0]

         for val in var.domain:
             yield from dfs_solver(rem_cons, context | {var: val},
                                   var_order[1:])
             
def dfs_solve_all(csp, var_order=None):

    if(var_order == None):
        var_order = list(csp.variables)

    return list(dfs_solver(csp.constraints, {}, var_order))

def dfs_solve1(csp, var_order=None):

    if(var_order == None):
        var_order = list(csp.variables)

    print(var_order)
    
    for sol in dfs_solver(csp.constraints, {}, var_order):
        return sol
