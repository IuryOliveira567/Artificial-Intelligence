import sys
from stripsProblem import STRIPS_domain, Strips, Planning_problem
from stripsForwardPlanner import Forward_STRIPS
from stripsRegressionPlanner import Regression_STRIPS
from stripsCSPPlanner import CSP_from_STRIPS, con_plan


sys.path.append('../SearchProblem')

from searchBranchAndBound import DF_branch_and_bound
from searchGeneric import Searcher
from SearchMPP import SearcherMPP
import stripsProblem
from stripsHeuristic import h1
from stripsPOP import POP_search_from_STRIPS


sys.path.append('../Constraint-Satisfaction-Problems')
from cspConsistency import Search_with_AC_from_CSP, Con_solver


def move(x, y, z):

    return 'move_' + x + '_from_' + y + '_to_' + z

def on(x):

    return x + '_is_on'

def clear(x):

    return 'clear_' + x

def create_blocks_world(blocks={'a', 'b', 'c', 'd'}):
    blocks_and_table = blocks | {'table'}

    stmap = {Strips(move(x, y, z), {on(x): y, clear(x): True, clear(z): True},
                    {on(x): z, clear(y): True, clear(z): False})
             for x in blocks
             for y in blocks_and_table
             for z in blocks
             if x != y and y != z and z != x}
    
    stmap.update({Strips(move(x, y, 'table'), {on(x): y, clear(x): True},
                         {on(x): 'table', clear(y): True})
            for x in blocks
            for y in blocks
            if x != y})

    feature_domain_dict = {on(x): blocks_and_table - {x} for x in blocks}
    feature_domain_dict.update({clear(x): boolean for x in blocks_and_table})

    return STRIPS_domain(feature_domain_dict, stmap)


if __name__ == "__main__":
   boolean = {False, True}
   
   delivery_domain = STRIPS_domain({'RLoc':{'cs', 'off', 'lab', 'mr'}, 'RHC':boolean, 'SWC':boolean,
         'MW':boolean, 'RHM':boolean}, #feature:values dictionary
        { Strips('mc_cs', {'RLoc':'cs'}, {'RLoc':'off'}),
          Strips('mc_off', {'RLoc':'off'}, {'RLoc':'lab'}),
          Strips('mc_lab', {'RLoc':'lab'}, {'RLoc':'mr'}),
          Strips('mc_mr', {'RLoc':'mr'}, {'RLoc':'cs'}),
          Strips('mcc_cs', {'RLoc':'cs'}, {'RLoc':'mr'}),
          Strips('mcc_off', {'RLoc':'off'}, {'RLoc':'cs'}),
          Strips('mcc_lab', {'RLoc':'lab'}, {'RLoc':'off'}),
          Strips('mcc_mr', {'RLoc':'mr'}, {'RLoc':'lab'}),
          Strips('puc', {'RLoc':'cs', 'RHC':False}, {'RHC':True}),
          Strips('dc', {'RLoc':'off', 'RHC':True}, {'RHC':False, 'SWC':False}),
          Strips('pum', {'RLoc':'mr','MW':True}, {'RHM':True,'MW':False}),
          Strips('dm', {'RLoc':'off', 'RHM':True}, {'RHM':False})
        })

   problem0 = Planning_problem(delivery_domain,
                               {'RLoc': 'lab', 'MW': True, 'SWC': True, 'RHC': False,
                                'RHM': False}, {'RLoc': 'off'})

   problem1 = Planning_problem(delivery_domain,
                               {'RLoc': 'lab', 'MW': True, 'SWC': True, 'RHC': False,
                                'RHM': False},
                               {'SWC': False})

   problem2 = Planning_problem(delivery_domain,
                               {'RLoc': 'lab', 'MW': True, 'SWC': True, 'RHC': False,
                                'RHM': False},
                               {'SWC': False, 'MW': False, 'RHM': False})

   blocks1dom = create_blocks_world({'a', 'b', 'c'})
   blocks1 = Planning_problem(blocks1dom,
                              {on('a'): 'table', clear('b'): True,
                               on('b'): 'c', clear('b'): True,
                               on('c'): 'table', clear('c'): False},
                              {on('a'): 'b', on('c'): 'a'})

   blocks2dom = create_blocks_world({'a', 'b', 'c', 'd'})
   tower4 = {clear('a'): True, on('a'): 'b',
             clear('b'): False, on('b'): 'c',
             clear('c'): False, on('c'): 'd',
             clear('d'): False, on('d'): 'table'}

   blocks2 = Planning_problem(blocks2dom, tower4,
                              {on('d'): 'c', on('c'): 'b', on('b'): 'a'})

   #paths = SearcherMPP(Forward_STRIPS(problem1)).search()
   #paths2 = DF_branch_and_bound(Forward_STRIPS(problem1), 10).search()
   #f = SearcherMPP(Forward_STRIPS(problem1, h1))

   #SearcherMPP(Regression_STRIPS(problem1)).search()
   #g = DF_branch_and_bound(Regression_STRIPS(problem2), 10).search()

   #con_plan(problem1, 5)
   #con_plan(problem1, 4)

   #searcher15a = Searcher(Search_with_AC_from_CSP(CSP_from_STRIPS(problem1,
   #                                                5))       

   rplanning0 = POP_search_from_STRIPS(problem1)
   searcher0 = DF_branch_and_bound(rplanning0, 5)






















  

   


   
