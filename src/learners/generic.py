from math import exp
import numpy as np


class QLearner(object):

    def __init__(self, env, discount=.9, init_lr=.1, lr_decay=.00001, lr_const=.001):
        self.env = env
        self.discount = discount
        self.init_lr = init_lr
        self.lr_const = lr_const
        self.lr_decay = lr_decay

        self.episodes = 0

        env.reset()

        self.Qs = [
            np.random.normal(
                loc=0., scale=3.,
                size=(env.STATE_SPACE, env.ACTION_SPACE, env.ACTION_SPACE))
            # np.ones((env.STATE_SPACE, env.ACTION_SPACE, env.ACTION_SPACE))
            for _ in range(2)
        ]

    def get_lr(self):
        return self.init_lr * exp(-self.lr_decay*self.episodes) + self.lr_const

    def _select_actions(self, state=None):
        return [0, 0]

    def _QtoV(self, state, player):
        return 0

    def _update_Qs(self, old_state, actions, next_state, rewards):
        for i in range(2):
            q = self.get_lr() * (rewards[i] + self.discount * self._QtoV(next_state, i))
            # if old_state.id == 71:
            #     print(old_state.id, actions, q)
            q += (1-self.get_lr())*self.Qs[i][old_state.id, actions[0], actions[1]]

            self.Qs[i][old_state.id, actions[0], actions[1]] = q

    def train(self):
        self.episodes += 1

        old_state = self.env.state
        actions = self._select_actions()
        next_state = self.env.step(actions)
        rewards = next_state.get_rewards()

        self._update_Qs(old_state, actions, next_state, rewards)

        if self.env.steps > 100:
            self.env.reset()
        if next_state.finished():
            self.env.reset()
            # print(rewards)


