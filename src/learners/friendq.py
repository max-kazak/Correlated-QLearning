import numpy as np

from .generic import QLearner


class FriendQ(QLearner):

    def __init__(self, env, discount=.9, init_lr=.1, lr_decay=.000005, lr_const=.001):
        super(FriendQ, self).__init__(env, discount, init_lr, lr_decay, lr_const)

    def _select_actions(self, state=None):
        s = state if state is not None else self.env.state
        actions = []
        # arr = self.Qs[0][s.id]
        # argmax = np.argmax(arr)
        # actions.append(argmax // arr.shape[1])
        # arr = self.Qs[1][s.id]
        # argmax = np.argmax(arr)
        # actions.append(argmax % arr.shape[1])
        arr = self.Qs[0][s.id]
        p = arr - np.min(arr)
        p = p / np.sum(p)
        p = p.flatten()
        a = np.random.choice(len(p), p=p)
        actions.append(a // arr.shape[1])

        arr = self.Qs[1][s.id]
        p = arr - np.min(arr)
        p = p / np.sum(p)
        p = p.flatten()
        a = np.random.choice(len(p), p=p)
        actions.append(a % arr.shape[1])

        return actions

    def _QtoV(self, state, player):
        v = np.max(self.Qs[player][state.id])
        return v
