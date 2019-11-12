import numpy as np
import cvxopt

from .generic import QLearner


# suppress output
cvxopt.solvers.options['show_progress'] = False
cvxopt.solvers.options['glpk'] = {'LPX_K_MSGLEV': 0, 'msg_lev': "GLP_MSG_OFF"}


class FoeQ(QLearner):

    def __init__(self, env, discount=.9, init_lr=.1, lr_decay=.000005, lr_const=.001):
        super(FoeQ, self).__init__(env, discount, init_lr, lr_decay, lr_const)

    def _select_actions(self, state=None):
        s = state if state is not None else self.env.state
        actions = []

        arr = self.Qs[0][s.id]
        p, _ = self._minmax_row(arr)
        a = np.random.choice(len(p), p=p)
        actions.append(a)
        # print("player 0: " + str(a) + " " + str(p))

        arr = self.Qs[1][s.id]
        p, _ = self._minmax_row(arr.T)
        a = np.random.choice(len(p), p=p)
        actions.append(a)
        # print("player 1: " + str(a) + " " + str(p))

        return actions

    def _minmax_row(self, values):
        arr = values.copy()

        # probability constraints
        G = np.identity(self.env.ACTION_SPACE + 1)[1:, :]
        G = G * -1

        # values constraints
        G = np.append(G, np.insert(arr.T, 0, -1, axis=1) * -1, axis=0)

        G = G.T

        # sum of all probabilities equal 1.
        A = np.ones((G.shape[0], 1))
        A[0, 0] = 0
        b = [[1.]]

        h = np.zeros(G.shape[1])

        # minimization target
        c = np.zeros(G.shape[0])
        c[0] = -1

        # run linear programming
        G = cvxopt.matrix(G.tolist())
        h = cvxopt.matrix(h.tolist())
        c = cvxopt.matrix(c.tolist())
        A = cvxopt.matrix(A.tolist())
        b = cvxopt.matrix(b)

        solution = cvxopt.solvers.lp(c, G, h, A, b, solver='glpk')

        solution = list(solution['x'])

        p = solution[1:]
        p = [pi if pi >= 0 else 0 for pi in p]
        sump = sum(p)
        p = [pi/sump for pi in p]

        return p, solution[0]  # return action probabilities and maximin value

    def _QtoV(self, state, player):
        arr = self.Qs[player][state.id]
        if player == 1:
            arr = arr.T
        p, v = self._minmax_row(arr)
        # w = (arr.T * p).T
        # sw = np.sum(w, axis=0)
        # min = np.min(sw)
        # print(v-min)
        return v
