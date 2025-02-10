def dist(loc1, loc2):

    if(loc1 == loc2):
        return 0

    if({loc1, loc2} in [{'cs', 'lab'}, {'mr', 'off'}]):
        return 2
    else:
        return 1

def h1(state, goal):

    if('RLoc' in goal):
        return dist(state['RLoc'], goal['RLoc'])
    else:
        return 0

def h2(state, goal):

    if('SWC' in goal and goal['SWC'] == False
       and state['SWC'] == True
       and state['RHC'] == False):

        return dist(state['Rloc'], 'cs') + 3
    else:
        return 0

def maxh(*heuristics):

    def newh(state, goal):

        return max(h(state, goal) for h in heuristic)

    return newh





