from logicProblem import *
from logicBottomUp import fixed_point
from logicTopDown import prove
from logicExplain import prove_atom, prove_body, interact
from logicAssumables import Assumable, KBA, minsets
from logicNegation import Not, prove_naf


triv_KB = KB([
    Clause('i_am', ['i_think']),
    Clause('i_think'),
    Clause('i_smell', ['i_exist'])
    ])

elect = KB([
  Clause('light_l1'),
  Clause('light_l2'),
  Clause('ok_l1'),
  Clause('ok_l2'),
  Clause('ok_cb1'),
  Clause('ok_cb2'),
  Clause('live_outside'),
  Clause('live_l1', ['live_w0']),
  Clause('live_w0', ['up_s2','live_w1']),
  Clause('live_w0', ['down_s2','live_w2']),
  Clause('live_w1', ['up_s1', 'live_w3']),
  Clause('live_w2', ['down_s1','live_w3' ]),
  Clause('live_l2', ['live_w4']),
  Clause('live_w4', ['up_s3','live_w3' ]),
  Clause('live_p_1', ['live_w3']),
  Clause('live_w3', ['live_w5', 'ok_cb1']),
  Clause('live_p_2', ['live_w6']),
  Clause('live_w6', ['live_w5', 'ok_cb2']),
  Clause('live_w5', ['live_outside']),
  Clause('lit_l1', ['light_l1', 'live_l1', 'ok_l1']),
  Clause('lit_l2', ['light_l2', 'live_l2', 'ok_l2']),
  Askable('up_s1'),
  Askable('down_s1'),
  Askable('up_s2'),
  Askable('down_s2'),
  Askable('up_s3'),
  Askable('down_s2')
])

def test(kb=triv_KB, fixedpt={'i_am', 'i_think'}):

    fp = fixed_point(kb)
    assert fp == fixedpt, f"kb gave result {fp}"
    print("Passed unit test")

if __name__ == "__main__":
   a1 = prove_atom(triv_KB, 'i_am')
   assert a1, f"triv_KB proving i_am gave{a1}"

   a2 = prove_atom(triv_KB, 'i_smell')
   assert a2 == "fail", "triv_KB proving i_smell gave {a2}"
   #print("Passed unit tests")

'''
c1 = Clause("A", ["B", "C"])
c2 = Clause("B", ["D", "E", "F"])
c3 = Clause("C", ["E", "C", "G"])
c4 = Clause("D", ["B", "C", "A"])

a1 = Assumable("A")
a2 = Askable("B")
a3 = Askable("C")
#a4 = Askable("D")
a5 = Askable("E")
a6 = Askable("F")
a7 = Askable("G")

triv_KB_naf = KB([
  Clause('i_am', ['i_think']),
  Clause('i_think'),
  Clause('i_smell', ['i_am', Not('dead')]),
  Clause('i_bad', ['i_am', Not('i_think')])
  ])

a1 = prove_naf(triv_KB_naf, ['i_smell'])
assert a1, f"triv_KB_naf proving i_smell gave {a1}"

a2 = prove_naf(triv_KB_naf, ['i_bad'])
assert not a2, f"triv_KB_naf proving i_bad gave {a2}"
'''

# Criando cláusulas
rules = [
    Clause("carro_não_funciona", ["bateria_vazia", "motor_quebrado"]),
    Clause("bateria_vazia", ["luz_fraca"]),
    Clause("motor_quebrado", ["não_gira"]),
    Clause("carro_não_funciona", ["combustível_acabou"]),
]

# Criando askables (perguntas para o usuário)
askables = [
    Askable("luz_fraca"),
    Askable("motor_gira"),
    Askable("combustível_acabou")
]

# Criando assumables (hipóteses)
assumables = [
    Assumable("bateria_boa"),
    Assumable("motor_bom")
]

# Criando a base de conhecimento
kb = KBA(rules + askables + assumables)
























