"""Transparent local baselines; these are not reproductions of published systems."""
import numpy as np
from controller_sequence import SequenceController

ACTIONS = (480, 544, 576, 640)


class Policy:
    def __init__(self, name, budget_ms=120, seed=0, controller=None):
        self.name, self.budget = name, budget_ms
        self.rng = np.random.default_rng(seed)
        self.index = 2
        self.good = 0
        self.cost = {}
        self.guarded=name.startswith('guarded')
        self.last_resolution=None
        self.dwell=0
        self.sequence = SequenceController.from_json(controller) if name == 'sequence' or self.guarded else None

    def choose(self, identifier):
        if self.name.startswith('fixed:'):
            return int(self.name.split(':')[1])
        if self.guarded:
            preferred=self.sequence.choose(identifier)
            feasible=[r for r in ACTIONS if r<=preferred and
                      self.cost.get(r,float('inf'))+(0 if self.name=='guarded-no-switch-cost' or r==self.last_resolution else 1.)<=self.budget*.9]
            chosen=max(feasible) if feasible else min(ACTIONS)
            if self.name!='guarded-no-hysteresis' and self.last_resolution in feasible and self.dwell<4:
                chosen=self.last_resolution
            return chosen
        if self.sequence:
            return self.sequence.choose(identifier)
        if self.name == 'random':
            return int(self.rng.choice(ACTIONS))
        if self.name == 'switch':
            return ACTIONS[int(identifier.split('-')[-1]) % len(ACTIONS)]
        if self.name == 'deadline':
            # Optimistic initialization explores all configurations. Afterwards
            # choose the largest whose EMA cost fits a 10% budget reserve.
            unexplored = [r for r in ACTIONS if r not in self.cost]
            if unexplored:
                return unexplored[0]
            feasible = [r for r in ACTIONS if self.cost[r] <= self.budget * .9]
            return max(feasible) if feasible else min(ACTIONS)
        if self.name == 'aimd':
            return ACTIONS[self.index]
        raise ValueError(f'unknown policy: {self.name}')

    def update(self, identifier, detections, resolution, service_ms):
        if self.sequence:
            self.sequence.update(identifier, detections)
        self.cost[resolution] = .25 * service_ms + .75 * self.cost.get(resolution, service_ms)
        # Refresh other scales using a shared contention ratio. This is a local
        # model-based baseline, not an implementation of ApproxDet or Chameleon.
        if self.name == 'deadline' or self.guarded:
            for r in list(self.cost):
                if r != resolution:
                    projected = service_ms * (r / resolution) ** 2
                    self.cost[r] = .1 * projected + .9 * self.cost[r]
        self.dwell=self.dwell+1 if resolution==self.last_resolution else 1
        self.last_resolution=resolution
        if self.name == 'aimd':
            if service_ms > self.budget:
                self.index = max(0, self.index - 1)
                self.good = 0
            elif service_ms < .8 * self.budget:
                self.good += 1
                if self.good >= 4:
                    self.index = min(len(ACTIONS) - 1, self.index + 1)
                    self.good = 0
            else:
                self.good = 0
