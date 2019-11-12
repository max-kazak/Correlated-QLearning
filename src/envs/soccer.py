import random

NROWS = 2
NCOLS = 4

ACTIONS = ['N', 'S', 'E', 'W', 'stick']
MOVES = [
    lambda y, x: (y-1 if y-1 >= 0 else y, x),
    lambda y, x: (y+1 if y+1 < NROWS else y, x),
    lambda y, x: (y, x+1 if x+1 < NCOLS else x),
    lambda y, x: (y, x-1 if x-1 >= 0 else x),
    lambda y, x: (y, x)
]


class FieldState(object):

    def __init__(self, coords, hasball, id=None):
        self.id = id
        self.coords = coords
        self.hasball = hasball

    def get_reward(self, player=0):
        if self.coords[self.hasball][1] == 0:
            return 100 if player == 0 else -100
        if self.coords[self.hasball][1] == NCOLS-1:
            return 100 if player == 1 else -100
        return 0

    def get_rewards(self):
        return [self.get_reward(0), self.get_reward(1)]

    def finished(self):
        if self.coords[self.hasball][1] in [0, NCOLS-1]:
            return True
        return False

    def __eq__(self, other):
        return other and self.coords == other.coords\
               and self.hasball == other.hasball

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.coords[0],
                     self.coords[1],
                     self.hasball))

    def __str__(self):
        map = [['#', '#', '#', '#'],
               ['#', '#', '#', '#']]
        psimbol = ['A' if self.hasball == 0 else 'a',
                   'B' if self.hasball == 1 else 'b']
        for player in range(2):
            coord = self.coords[player]
            map[coord[0]][coord[1]] = psimbol[player]
        return '\n'.join([''.join(l) for l in map])

    def __repr__(self):
        return self.__str__()


class Soccer(object):

    def __init__(self, debug=False):
        self.debug = debug
        self._gen_states()
        self.reset()

        self.STATE_SPACE = len(self.states)
        self.ACTION_SPACE = len(ACTIONS)

    def _gen_states(self):
        self.states = dict()
        id = 0
        for has_ball in range(2):
            for ai in range(NROWS):
                for aj in range(NCOLS):
                    for bi in range(NROWS):
                        for bj in range(NCOLS):
                            if (ai, aj) != (bi, bj):
                                s = FieldState([(ai, aj), (bi, bj)],
                                                           has_ball,
                                                           id)
                                self.states[s] = s
                                id += 1

    def reset(self):
        state = FieldState([(0, 2), (0, 1)], 1)
        self.state = self.states[state]
        self.steps = 0

    def step(self, actions):
        if self.debug:
            print(self.steps, ": ", actions)

        self.steps += 1
        move_order = random.choice([[0, 1], [1, 0]])

        coords = list(self.state.coords)
        hasball = self.state.hasball

        for player in move_order:
            # make a move one by one
            a = actions[player]
            new_coord = MOVES[a](*coords[player])
            if new_coord == coords[other(player)]:
                new_coord = coords[player]  # don't move
                if hasball == player:
                    hasball = other(player)  # loses ball
            coords[player] = new_coord

        state = FieldState(coords, hasball)
        self.state = self.states[state]

        if self.debug:
            print(self.state)

        return self.state


def other(player):
    return 0 if player == 1 else 1

if __name__ == "__main__":
    coord = (0, 1)
    for i in range(5):
        print(MOVES[i](*coord))
    env = Soccer()
    print(len(env.states))
    print(env.state)
    print(env.state.id)
    env.step([3, 2])
