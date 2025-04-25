from cspConsistency import Con_solver
import matplotlib.pyplot as plt


class ConsistencyGUI(Con_solver):

    def __init__(self, csp, fontsize=10, speed=1, **kwargs):

        self.fontsize = fontsize
        self.delay_time = 1 / speed

        Con_solver.__init__(self, csp, **kwargs)
        csp.show(showAutoAC = True)

    def go(self):

        res = self.solve_all()
        self.csp.draw_graph(domains=self.domains,
                            title="No more solutions. Gui finished.",
                            fontsize=self.fontsize)

        return res

    def _select_arc(self, to_do):

        while(True):

            self.csp.draw_graph(domains=self.domains, to_do=to_do,
                                title="Click on to_do (blue) arc",
                                fontsize=self.fontsize)
    
            while(self.csp.picked == None and not self.csp.autoAC):
                plt.pause(0.01)

            if(self.csp.autoAC):
                break

            picked = self.csp.picked
            self.csp.picked = None

            if(picked in to_do):
                to_do.remove(picked)
                print(f"{picked} picked")
                return picked
            else:
                print(f"{picked} not in to_do")

        if(self.csp.autoAC):
            self.csp.draw_graph(domains=self.domains, to_do=to_do,
                                title="Auto AC", fontsize=self.fontsize)
            plt.pause(self.delay_time)
            return to_do.pop()

    def select_var(self, iter_vars):

        vars = list(iter_vars)

        while(True):
            self.csp.draw_graph(domains=self.domains,
                                title="Arc consistent. Click node to split",
                                fontsize=self.fontsize)

            while(self.csp.picked == None):
                plt.pause(0.01)

            picked = self.csp.picked
            self.csp.picked = None
            self.csp.autoAC = False

            if(picked in vars):
                return picked
            else:
                print(picked, "not in", vars)

    def display(self, n, *args, **nargs):

        if(n <= self.max_display_level):
            print(*args, **nargs)

        if(n == 1):
            self.csp.draw_graph(domains=self.domains, to_do=set(),
                                title=' '.join(args) + ": click any node or arc to continue",
                                fontsize=self.fontsize)

            self.csp.autoAC = False

            while(self.csp.picked == None and not self.csp.autoAC):
                plt.pause(0.01)

            self.csp.picked = None
        elif(n == 2):
            plt.title("backtracking : click any node or arc to continue")
            self.csp.autoAC = False

            while(self.csp.picked == None and not self.csp.autoAC):
                plt.pause(0.01)

            self.csp.picked = None
        elif(n == 3):
            line = self.csp.thelines[self.arc_selected]
            line.set_color('red')
            line.set_linewidth(10)
            plt.pause(self.delay_time)
            line.set_color('limegreen')
            line.set_linewidth(self.csp.linewidth)
