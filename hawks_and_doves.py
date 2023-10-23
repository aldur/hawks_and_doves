#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from bisect import bisect
import collections
import dataclasses
from enum import Enum, IntEnum
import enum
from itertools import accumulate, groupby
import random

import matplotlib.pyplot as plt
import pandas as pd

#|%%--%%| <ZpLQjvpAX6|R1vGC6acZj>

# Assumption: There's no way to know someone else's strategy without figthing.

# |%%--%%| <R1vGC6acZj|SeMsFNNqLE>


class Strategy(Enum):
    HAWK = enum.auto()
    DOVE = enum.auto()
    RANDOM_HAWK_BIASED = enum.auto()
    RETALIATOR = enum.auto()

    def __repr__(self) -> str:
        return self._name_


class Points(IntEnum):
    WIN = 50
    LOSS = 0
    SERIOUS_INJURY = -100
    WASTE_OF_TIME = -10

    def __repr__(self) -> str:
        return f"{self.value}"


@dataclasses.dataclass
class Result:
    a_score: int
    b_score: int

    def reverse(self):
        self.a_score, self.b_score = self.b_score, self.a_score
        return self

    def shuffle(self):
        if random.random() < 0.5:  # random.binomialvariate
            self.reverse()
        return self


def fight(a: Strategy, b: Strategy) -> Result:
    match (a, b):
        case (Strategy.DOVE, Strategy.DOVE):
            return Result(
                Points.WIN + Points.WASTE_OF_TIME, Points.WASTE_OF_TIME
            ).shuffle()
        case (Strategy.DOVE, Strategy.HAWK):
            return Result(Points.LOSS, Points.WIN)
        case (Strategy.HAWK, Strategy.DOVE):
            return fight(b, a).reverse()
        case (Strategy.HAWK, Strategy.HAWK):
            return Result(Points.SERIOUS_INJURY, Points.WIN).shuffle()
        case (Strategy.RANDOM_HAWK_BIASED, _):
            strategy = Strategy.HAWK if random.random() < 7 / 12 else Strategy.DOVE
            return fight(strategy, b)
        case (_, Strategy.RANDOM_HAWK_BIASED):
            return fight(b, a).reverse()
        case (Strategy.RETALIATOR, Strategy.RETALIATOR):
            return fight(Strategy.DOVE, Strategy.DOVE)
        case (Strategy.RETALIATOR, _):
            return fight(b, b)
        case (_, Strategy.RETALIATOR):
            return fight(a, a)
        case _:
            assert False


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


def simulate(
    population, n_fights=1000
) -> tuple[dict[Strategy, dict[Strategy, float]], dict[Strategy, int]]:
    split = collections.Counter(population)

    indexes = [i for i, _ in enumerate(population)]

    # TODO: Cross-product to get species vs species!
    scores_by_species = collections.defaultdict(list)
    for _ in range(n_fights):
        idx_a, idx_b = random.sample(indexes, 2)
        a, b = population[idx_a], population[idx_b]

        result = fight(a, b)

        scores_by_species[a].append((b, result.a_score))
        scores_by_species[b].append((a, result.b_score))

    def key(t):
        species, _ = t
        return species.value

    scores_matrix = {}
    split = dict(split)  # Prevent getting 0 if asking for a missing value.
    for species, scores in scores_by_species.items():
        sorted_against_species = sorted(scores, key=key)
        grouped = groupby(sorted_against_species, key=key)
        averaged_by_group = list(
            (s, mean(score for (_, score) in g)) for s, g in grouped
        )
        scores_matrix[species] = {Strategy(s): avg for s, avg in averaged_by_group}

    return scores_matrix, split


# |%%--%%| <SeMsFNNqLE|r0ej7vJAoa>

population = [Strategy.DOVE for _ in range(100)]
scores_matrix, split = simulate(population)
split = pd.Series(split)
print(split)
scores_matrix = pd.DataFrame.from_dict(scores_matrix, orient="index")
print(scores_matrix)

# |%%--%%| <r0ej7vJAoa|mJ6PD3diHn>

population = [Strategy.DOVE for _ in range(100)]
population[0] = Strategy.HAWK
population[1] = Strategy.HAWK
scores_matrix, split = simulate(population)
split = pd.Series(split)
print(split)
scores_matrix = pd.DataFrame.from_dict(scores_matrix, orient="index")
print(scores_matrix)

# |%%--%%| <mJ6PD3diHn|9PfqgtKOiY>

population = [Strategy.HAWK for _ in range(100)]
scores_matrix, split = simulate(population)
split = pd.Series(split)
print(split)
scores_matrix = pd.DataFrame.from_dict(scores_matrix, orient="index")
print(scores_matrix)

# |%%--%%| <9PfqgtKOiY|MM3tv3dWuA>

population_size = 10000
n_generations = 125
results = []
population = [Strategy.HAWK] * (population_size // 2) + [Strategy.DOVE] * (
    population_size // 2
)


def scale(d, min_payoff, max_payoff):
    # negative numbers to 0..1
    # positive numbers to 1..2
    assert min_payoff < 0
    if min_payoff < 0 and abs(min_payoff) > max_payoff:
        max_payoff = abs(min_payoff)

    r = {k: (v - min_payoff) / (0 - min_payoff) for k, v in d.items() if v < 0} | {
        k: (1 + v - 0) / (max_payoff - 0) for k, v in d.items() if v > 0
    }
    assert all(0 <= v <= 2 for v in r.values())
    return r


for i in range(n_generations):
    scores_matrix, split = simulate(population, n_fights=10000)
    print(f"Split: {split}")
    results.append(split)

    weighted_average_scores = {}
    for species, averaged_scores_by_species in scores_matrix.items():
        weighted_average_scores[species] = mean(
            avg * split[s] for s, avg in averaged_scores_by_species.items()
        ) / len(population)

    print("Weighted average scores: ")
    print(weighted_average_scores)

    # TODO: Min/max payoff instead?
    # min_payoff = min(min(v for v in s.values()) for s in scores_matrix.values())
    # max_payoff = max(max(v for v in s.values()) for s in scores_matrix.values())
    min_payoff = min(p.value for p in Points)
    max_payoff = max(p.value for p in Points)

    scaled_weighted_average_scores = scale(
        weighted_average_scores, min_payoff, max_payoff
    )
    print("Scaled weighted average scores: ")
    print(scaled_weighted_average_scores)

    absolute_fitness = {
        k: split[k] * v / population_size
        for k, v in scaled_weighted_average_scores.items()
    }

    population = random.choices(
        tuple(absolute_fitness.keys()),
        weights=tuple(absolute_fitness.values()),
        k=population_size,
    )

    print()

# |%%--%%| <MM3tv3dWuA|RZVASJHVo9>

df = pd.DataFrame(results) / population_size
df.plot.line()
plt.axhline(y=(7 / 12), color="r", linestyle="-")
plt.xlabel("Generation")
plt.ylabel("% of individuals")

# |%%--%%| <RZVASJHVo9|0OF5YOLY0D>

population = [Strategy.DOVE, Strategy.HAWK]

weights = [0.69, 0.72]
cum_weights = list(accumulate(weights))
total = cum_weights[-1] + 0.0
hi = len(population) - 1
bisect(cum_weights, random.random() * total, 0, hi)
