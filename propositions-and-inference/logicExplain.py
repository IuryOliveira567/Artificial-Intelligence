from logicProblem import yes


def prove_atom(kb, atom, indent=""):

    kb.display(2, indent, 'proving', atom)

    if(atom in kb.askables):
        if(yes(input("Is " + atom + " true? "))):
           return (atom, "answered")
        else:
            return "fail"
    else:
        for cl in kb.clauses_for_atom(atom):
            kb.display(2, indent, "trying", atom, '<-', ' & '.join(cl.body))
            pr_body = prove_body(kb, cl.body, indent)

            if(pr_body != "fail"):
                return (atom, pr_body)

        return "fail"

def prove_body(kb, ans_body, indent=""):

    proofs = []

    for atom in ans_body:
        proof_at = prove_atom(kb, atom, indent + " ")
        if(proof_at == "fail"):
            return "fail"
        else:
            proofs.append(proof_at)
            
    return proofs

helptext = """Commands are: ask atom ask is there is a proof for atom (atom should not be in quotes)
              how show the clause that was used to prove atom
              how n show the clause used to prove the nth element of the body
              up go back up proof tree to explore other parts of the proof tree
              kb print the knowledge base
              quit quit this interaction (and go back to Python)
              help print this text
"""

def interact(kb):

    going = True
    ups = []
    proof = "fail"

    while(going):
        inp = input("logic Explain : ")
        inps = inp.split(" ")

        try:
            command = inps[0]
            if(command == "quit"):
                going = False
            elif(command == "ask"):
                proof = prove_atom(kb, inps[1])

                if(proof == "fail"):
                    print("fail")
                else:
                    print("yes")
            elif(command == "how"):
                if(proof == "fail"):
                    print("there is no proof")
                elif(len(inps) == 1):
                    print_rule(proof)
                else:
                    try:
                        ups.append(proof)
                        proof = proof[1][int(inps[1])]
                        print_rule(proof)
                    except:
                        print('In "how n", n must be a number between 0 and', len(proof[1]) -1, "inclusive.")
            elif(command == "up"):
                if(ups):
                    proof = ups.pop()
                else:
                    print("No rule to go up to.")

                print_rule(proof)
            elif(command == "kb"):
                print(helptext)
            else:
                print("unkown command: ", inp)
                print("use help for help")
        except:
            print("unkown command : ", inp)
            print("use help for help")

def print_rule(proof):
    (head, body) = proof

    if(body == "answered"):
        print(head, "was answered yes")
    elif(body == []):
        print(head, "is a fact")
    else:
        print(head, "<-")
        for i, a in enumerate(body):
            print(i, ":", a[0])
