from logicProblem import yes


def prove(kb, ans_body, indent=""):

    kb.display(2, indent, 'yes <-',' & '.join(ans_body))

    if(ans_body):
        selected = ans_body[0]
        if(selected in kb.askables):
            return (yes(input("Is " + selected + " true? "))
                    and prove(kb, ans_body[1:], indent + " "))
        else:
            return any(prove(kb, cl.body + ans_body[1:], indent + " ")
                       for cl in kb.clauses_for_atom(selected))
    else:
        return True
