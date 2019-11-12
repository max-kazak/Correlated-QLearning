import random
from math import exp

import numpy as np

from .generic import QLearner


class BasicQ(QLearner):

    def __init__(self, env, discount=.9, init_lr=.1, lr_decay=.000005, lr_const=.001, init_eps=1., eps_decay=.00001, eps_const=.001):
        super(BasicQ, self).__init__(env, discount, init_lr, lr_decay, lr_const)

        self.init_eps = init_eps
        self.eps_decay = eps_decay
        self.eps_const = eps_const

        self.Qs = [
            np.random.normal(size=(env.STATE_SPACE, env.ACTION_SPACE))
            for _ in range(2)
        ]

    def get_eps(self):
        return self.init_eps * exp(-self.eps_decay*self.episodes) + self.eps_const

    def _update_Qs(self, old_state, actions, next_state, rewards):
        for i in range(2):
            q = (1-self.get_lr())*self.Qs[i][old_state.id, actions[i]]
            q += self.get_lr() * (rewards[i] + self.discount*self._QtoV(next_state, i))
            self.Qs[i][old_state.id, actions[i]] = q

    def _select_actions(self, state=None):
        s = state if state is not None else self.env.state
        actions = []
        for i in range(2):
            if random.random() < self.get_eps():
                a = random.choice(range(self.env.ACTION_SPACE))
            else:
                a = np.argmax(self.Qs[i][s.id])
            actions.append(a)

        return actions

    def _QtoV(self, state, player):
        v = np.max(self.Qs[player][state.id,:])
        return v
