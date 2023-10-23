#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import dataclasses
from enum import Enum, IntEnum, auto
from itertools import groupby
import random

import pandas as pd

#|%%--%%| <sn894epeob|wlXlRnL3RM>

# Assumption: There's no way to know someone else's strategy without figthing.

# |%%--%%| <wlXlRnL3RM|SeMsFNNqLE>


class Strategy(Enum):
    HAWK = auto()
    DOVE = auto()

    # make prettier prints
    def __repr__(self) -> str:
        return self._name_


class Points(IntEnum):
    WIN = 50
    LOSS = 0
    SERIOUS_INJURY = -100
    WASTE_OF_TIME = -10

    # make prettier prints
    def __repr__(self) -> str:
        return f"{self.value}"


@dataclasses.dataclass
class FightResult:
    a_score: int
    b_score: int

    def reverse(self):
        self.a_score, self.b_score = self.b_score, self.a_score
        return self

    def shuffle(self):
        if random.random() < 0.5:
            self.reverse()
        return self


# |%%--%%| <SeMsFNNqLE|TRVmKtFFDY>


def fight(a: Strategy, b: Strategy) -> FightResult:
    match (a, b):
        case (Strategy.DOVE, Strategy.DOVE):
            return FightResult(
                Points.WIN + Points.WASTE_OF_TIME, Points.WASTE_OF_TIME
            ).shuffle()
        case (Strategy.DOVE, Strategy.HAWK):
            return FightResult(Points.LOSS, Points.WIN)
        case (Strategy.HAWK, Strategy.DOVE):
            return fight(b, a).reverse()
        case (Strategy.HAWK, Strategy.HAWK):
            return FightResult(Points.SERIOUS_INJURY, Points.WIN).shuffle()
        case _:
            assert False


# |%%--%%| <TRVmKtFFDY|yqJmbsdYAS>


def mean(data) -> float:
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
    n = 0
    mean = 0.0

    for x in data:
        n += 1
        mean += (x - mean) / n

    if n < 1:
        assert False
        return float("nan")
    else:
        return mean


# |%%--%%| <yqJmbsdYAS|R1vGC6acZj>


def simulate(population, n_fights=1000) -> dict[Strategy, dict[Strategy, float]]:

    scores_by_species = collections.defaultdict(list)
    for _ in range(n_fights):
        a, b = random.sample(population, k=2)  # without replacement

        result = fight(a, b)

        scores_by_species[a].append((b, result.a_score))
        scores_by_species[b].append((a, result.b_score))

    def key(t):
        species, _ = t
        return species.value

    scores_matrix = {}
    for species, scores in scores_by_species.items():
        sorted_against_species = sorted(scores, key=key)
        grouped = groupby(
            sorted_against_species, key=key
        )  # needs elements sorted by group key
        scores_matrix[species] = {
            Strategy(s): mean(score for (_, score) in g) for s, g in grouped
        }  # the matrix row for `species`

    return scores_matrix


# |%%--%%| <R1vGC6acZj|r0ej7vJAoa>

population = [Strategy.DOVE for _ in range(100)]
split = collections.Counter(population)
scores_matrix = simulate(population)
print(split)
scores_matrix = pd.DataFrame.from_dict(scores_matrix, orient="index")
print(scores_matrix)

#|%%--%%| <r0ej7vJAoa|mUdf1JF6HP>

population = [Strategy.DOVE for _ in range(100)]
population[0] = Strategy.HAWK
population[1] = Strategy.HAWK
split = collections.Counter(population)
scores_matrix = simulate(population)
print(split)
scores_matrix = pd.DataFrame.from_dict(scores_matrix, orient="index")
print(scores_matrix)

# |%%--%%| <mUdf1JF6HP|9PfqgtKOiY>
population = [Strategy.HAWK for _ in range(100)]
split = collections.Counter(population)
scores_matrix = simulate(population)
print(split)
scores_matrix = pd.DataFrame.from_dict(scores_matrix, orient="index")
print(scores_matrix)


# |%%--%%| <9PfqgtKOiY|mJ6PD3diHn>

population_size = 100
population = [Strategy.DOVE] * (population_size // 2) + [Strategy.HAWK] * (population_size // 2)
assert len(population) == population_size
split = collections.Counter(population)
print(split)
scores_matrix = simulate(population)
scores_matrix = pd.DataFrame.from_dict(scores_matrix, orient="index")
print(scores_matrix)
