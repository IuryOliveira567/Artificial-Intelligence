from variable import Variable
from cspProblem import CSP, Constraint


def queens(ri, rj):

    def no_take(ci, cj):

        return ci != cj and abs(ri - ci) != abs(rj - cj)

    return no_take

def n_queens(n):

    columns = list(range(n))
    variables = [Variable(f"R{i}", columns) for i in range(n)]
    
    return CSP("n-queens",
               variables,
               [Constraint([variables[i], variables[j]], queens(i, j))
                for i in range(n) for j in range(n) if i != j])
