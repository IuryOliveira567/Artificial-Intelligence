from logicProblem import Clause, Askable, KB, yes


class Assumable(object):

    def __init__(self, atom):

        self.atom = atom

    def __str__(self):

        return "assumable " + self.atom + "."

class KBA(KB):

    def __init__(self, statements):
        self.assumables = [c.atom for c in statements if isinstance(c, Assumable)]
        KB.__init__(self, statements)

    def prove_all_ass(self, ans_body, assumed=set()):

        if(ans_body):
            selected = ans_body[0]
            if(selected in self.askables):
                if(yes(input("Is " + selected + " true? "))):
                    return self.prove_all_ass(ans_body[1:], assumed)
                else:
                    return []
            elif(selected in self.assumables):
                return self.prove_all_ass(ans_body[1:], assumed | {selected})
            else:
                return [ass for cl in self.clauses_for_atom(selected)
                        for ass in self.prove_all_ass(cl.body + ans_body[1:], assumed)]
        else:
            return [assumed]

    def conflicts(self):

        return minsets(self.prove_all_ass(['false']))

def minsets(ls):

    ans = []
    for c in ls:
        if(not any(c1 < c for c1 in ls) and not any(c1 <= c for c1 in ans)):
            ans.append(c)

    return ans

def diagnose(cons):

    if(cons == []):
        return [set()]
    else:
        return minsets([({e} | d)
                        for e in cons[0]
                        for d in diagnoses(cons[1:])])
