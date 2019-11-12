import numpy as np
import cvxopt

from .generic import QLearner


# suppress output
cvxopt.solvers.options['show_progress'] = False
cvxopt.solvers.options['glpk'] = {'LPX_K_MSGLEV': 0, 'msg_lev': "GLP_MSG_OFF"}


class CEQ(QLearner):

    def __init__(self, env, discount=.9, init_lr=.1, lr_decay=.000005, lr_const=.001, solver='glpk'):
        super(CEQ, self).__init__(env, discount, init_lr, lr_decay, lr_const)
        self.solver = solver

    def _select_actions(self, state=None):
        s = state if state is not None else self.env.state
        actions = []

        p, _ = self.ce(s.id)
        if p is None:
            p = np.ones((self.env.ACTION_SPACE, self.env.ACTION_SPACE))
            p = p / 25
        p = p.flatten()
        a = np.random.choice(len(p), p=p)
        actions.append(a // self.env.ACTION_SPACE)
        # print("player 0: " + str(a) + " " + str(p))

        a = np.random.choice(len(p), p=p)
        actions.append(a % self.env.ACTION_SPACE)
        # print("player 1: " + str(a) + " " + str(p))

        return actions

    def ce(self, state_id):
        # add probability constraints (p>=0)
        G = np.identity(self.env.ACTION_SPACE ** 2)
        G = G * -1

        # add constraints for row player
        arr = self.Qs[0][state_id]
        for p in range(self.env.ACTION_SPACE):  # primal action row
            for s in range(self.env.ACTION_SPACE):  # secondary other choice row
                if p != s:
                    diff = arr[p] - arr[s]
                    constr = np.zeros(self.env.ACTION_SPACE ** 2)
                    for c in range(diff.shape[0]):
                        constr[p * self.env.ACTION_SPACE + c] = diff[c]
                    constr = constr * -1
                    constr = np.array([constr])
                    G = np.append(G, constr, axis=0)

        # add constraints for column player
        arr = self.Qs[1][state_id].T
        for p in range(self.env.ACTION_SPACE):  # primal action row
            for s in range(self.env.ACTION_SPACE):  # secondary other choice row
                if p != s:
                    diff = arr[p] - arr[s]
                    constr = np.zeros(self.env.ACTION_SPACE ** 2)
                    for c in range(diff.shape[0]):
                        constr[c * self.env.ACTION_SPACE + p] = diff[c]
                    constr = constr * -1
                    constr = np.array([constr])
                    G = np.append(G, constr, axis=0)

        G = G.T  # flip G for cvxopt

        # print(G.shape)  # returns 65 constraints + 1 equality constraint = 67 inequality constraints

        h = np.zeros(G.shape[1])

        # minimization target
        c = self.Qs[0][state_id].flatten() + self.Qs[1][state_id].flatten()
        c = c * -1

        # all probabilities should sum to 1
        A = np.ones((G.shape[0], 1))
        b = [[1.]]

        # find solution
        G = cvxopt.matrix(G.tolist())
        h = cvxopt.matrix(h.tolist())
        c = cvxopt.matrix(c.tolist())
        A = cvxopt.matrix(A.tolist())
        b = cvxopt.matrix(b)

        solution = cvxopt.solvers.lp(c, G, h, A, b, solver=self.solver)

        if solution['primal objective'] is None:
            print("oops can't compute LP")
            return None, None

        p = list(solution['x'])

        p = [pi if pi >= 0 else 0 for pi in p]
        sump = sum(p)
        p = [pi/sump for pi in p]

        p = np.array(p)
        p = np.reshape(p, (self.env.ACTION_SPACE, self.env.ACTION_SPACE))

        return p, solution['primal objective']  # return action probabilities and maximin value

    def _QtoV(self, state):
        vs = []
        p, v = self.ce(state.id)
        if p is None:
            return None
        for i in range(2):
            arr = self.Qs[i][state.id]
            v = np.sum(p * arr)
            vs.append(v)
        return vs

    def _update_Qs(self, old_state, actions, next_state, rewards):
        Vs = self._QtoV(next_state)
        if Vs is None:
            return
        for i in range(2):
            q = self.get_lr() * (rewards[i] + self.discount * Vs[i])
            # if old_state.id == 71:
            #     print(old_state.id, actions, q)
            q += (1-self.get_lr())*self.Qs[i][old_state.id, actions[0], actions[1]]

            self.Qs[i][old_state.id, actions[0], actions[1]] = q
