from variable import Variable
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from operator import lt, ne, eq, gt


class Constraint(object):

    def __init__(self, scope, condition, string=None, position=None):

        self.scope = scope
        self.condition = condition

        if(string is None):
            self.string = f"{self.condition.__name__}({self.scope})"
        else:
            self.string = string

        self.position = position

    def __repr__(self):

        return self.string

    def can_evaluate(self, assignment):

        return all(v in assignment for v in self.scope)

    def holds(self, assignment):    
            
        return self.condition(*tuple(assignment[v] for v in self.scope))

class CSP(object):

    def __init__(self, title, variables, constraints):

        self.title = title
        self.variables = variables
        self.constraints = constraints
        self.var_to_const = {var: set() for var in self.variables}

        for con in constraints:
            for var in con.scope:
                self.var_to_const[var].add(con)

    def __str__(self):

        return str(self.title)

    def __repr__(self):

        return f"CSP({self.title}, {self.variables}, {([str(c) for c in self.constraints])})"

    def consistent(self, assignment):
            
        return all(con.holds(assignment) for con in self.constraints if con.can_evaluate(assignment))

    def show(self, linewidth=3, showDomains=False, showAutoAC=False):

        self.linewidth = linewidth
        self.picked = None
        plt.ion()

        self.arcs = {}
        self.thelines = {}
        self.nodes = {}
        self.fig, self.ax = plt.subplots(1, 1)
        self.ax.set_axis_off()

        for var in self.variables:
            if(var.position is None):
                var.position = (random.random(), random.random())

        self.showAutoAC = showAutoAC

        domains = {var: var.domain for var in self.variables} if showDomains else {}
        self.draw_graph(domains=domains)

    def draw_graph(self, domains={}, to_do={}, title=None, fontsize=10):

        self.ax.clear()
        self.ax.set_axis_off()

        if(title):
            plt.title(title, fontsize=fontsize)
        else:
            plt.title(self.title, fontsize=fontsize)

        var_bbox = dict(boxstyle="round4,pad=1.0,rounding_size=0.5")
        con_bbox = dict(boxstyle="square,pad=1.0", color="green")

        self.autoACtext = plt.text(0, 0, "Auto AC" if self.showAutoAC else "",
                                   bbox={'boxstyle': 'square', 'color': 'yellow'},
                                   picker=True, fontsize=fontsize)

        for con in self.constraints:
            if(con.position is None):
                con.position = tuple(sum(var.position[i] for var in con.scope) / len(con.scope)
                                     for i in range(2))

            cx, cy = con.position
            bbox = dict(boxstyle="square,pad=1.0", color="green")

            for var in con.scope:
                vx, vy = var.position
                if(var, con) in to_do:
                    color = "blue"
                else:
                    color = "limegreen"

                line = lines.Line2D([cx, vx], [cy, vy], axes=self.ax,
                                    color=color, picker=True, pickradius=10,
                                    linewidth=self.linewidth)

                self.arcs[line] = (var, con)
                self.thelines[(var, con)] = line
                self.ax.add_line(line)

            plt.text(cx, cy, con.string, bbox=con_bbox, ha='center', va='center', fontsize=fontsize)

            for var in self.variables:
                x, y = var.position

                if(domains):
                    node_label = f"{var.name}\n{domains[var]}"
                else:
                    node_label = var.name

                node = plt.text(x, y, node_label, bbox=var_bbox, ha='center',
                                va='center', picker=True, fontsize=fontsize)

                self.nodes[node] = var

            self.fig.canvas.mpl_connect('pick_event', self.pick_handler)

    def pick_handler(self, event):

        mouseevent = event.mouseevent
        self.last_artist = artist = event.artist

        if(artist in self.arcs):
            self.picked = self.arcs[artist]
        elif(artist in self.nodes):
            self.picked = self.nodes[artist]
        elif(artist == self.autoACtext):
            self.autoAC = True
        else:
            print("### unkown click")

def ne_(val):

    def nev(x):
        return val != x

    nev.__name__ = f"{val} != "
    return nev

def is_(val):

    def isv(x):
        return val == x

    isv.__name__ = f"{val} == "
    return isv
