from logicProblem import KB, Clause, Askable, yes


class Not(object):

    def __init__(self, atom):

        self.theatom = atom

    def atom(self):

        return self.theatom

    def __repr__(self):

        return f"Not({self.theatom})"
    
def prove_naf(kb, ans_body, indent=""):

    kb.display(2, indent, 'yes <-', ' & '.join(str(e) for e in ans_body))

    if(ans_body):
        selected = ans_body[0]

        if(isinstance(selected, Not)):
           kb.display(2, indent, f"proving {selected.atom()}")

           if(prove_naf(kb, [selected.atom()], indent)):
              kb.display(2, indent, f"{selected.atom()} succeeded so Not({selected.atom()}) fails")
              return False
           else:
               kb.display(2, indent, f"{selected.atom()} fails so Not({selected.atom()}) succeeds")
               return prove_naf(kb, ans_body[1:], indent + " ")

        if(selected in kb.askables):
            return (yes(input("Is " + selected + " true? ")) and prove_naf(kb, ans_body[1:], indent + " "))
        else:
            return any(prove_naf(kb, cl.body + ans_body[1:], indent + " ") for cl in kb.clauses_for_atom(selected))
    else:
        return True
