from logicProblem import yes


def fixed_point(kb):

    fp = ask_askables(kb)
    added = True

    while(added):
        added = False

        for c in kb.clauses:
            if(c.head not in fp and all(b in fp for b in c.body)):
              fp.add(c.head)
              added = True
              kb.display(2, c.head, "added to fp due to clause", c)

    return fp

def ask_askables(kb):

    return {at for at in kb.askables if yes(input("Is " + at + " true? "))}
