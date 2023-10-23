#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import dataclasses
from enum import Enum, IntEnum, auto
from itertools import groupby
import math
import random

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme()

# |%%--%%| <sn894epeob|wlXlRnL3RM>

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
    a_payoff: int
    b_payoff: int

    def reverse(self):
        self.a_payoff, self.b_payoff = self.b_payoff, self.a_payoff
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

    payoffs_by_species = collections.defaultdict(list)
    for _ in range(n_fights):
        a, b = random.sample(population, k=2)  # without replacement

        result = fight(a, b)

        payoffs_by_species[a].append((b, result.a_payoff))
        payoffs_by_species[b].append((a, result.b_payoff))

    def key(t):
        species, _ = t
        return species.value

    payoffs_matrix = {}
    for species, payoffs in payoffs_by_species.items():
        sorted_against_species = sorted(payoffs, key=key)
        grouped = groupby(
            sorted_against_species, key=key
        )  # needs elements sorted by group key
        payoffs_matrix[species] = {
            Strategy(s): mean(payoff for (_, payoff) in g) for s, g in grouped
        }  # the matrix row for `species`

    return payoffs_matrix


# |%%--%%| <R1vGC6acZj|r0ej7vJAoa>

population = [Strategy.DOVE for _ in range(100)]
split = collections.Counter(population)
payoffs_matrix = simulate(population)
print(split)
payoffs_matrix = pd.DataFrame.from_dict(payoffs_matrix, orient="index")
print(payoffs_matrix)

# |%%--%%| <r0ej7vJAoa|mUdf1JF6HP>

population = [Strategy.DOVE for _ in range(100)]
population[0] = Strategy.HAWK
population[1] = Strategy.HAWK
split = collections.Counter(population)
payoffs_matrix = simulate(population)
print(split)
payoffs_matrix = pd.DataFrame.from_dict(payoffs_matrix, orient="index")
print(payoffs_matrix)

# |%%--%%| <mUdf1JF6HP|9PfqgtKOiY>

population = [Strategy.HAWK for _ in range(100)]
split = collections.Counter(population)
payoffs_matrix = simulate(population)
print(split)
payoffs_matrix = pd.DataFrame.from_dict(payoffs_matrix, orient="index")
print(payoffs_matrix)

# |%%--%%| <9PfqgtKOiY|0wmVWPdvIj>

population_size = 100
population = [Strategy.DOVE] * (population_size // 2) + [Strategy.HAWK] * (
    population_size // 2
)
assert len(population) == population_size
split = collections.Counter(population)
print(split)
payoffs_matrix = simulate(population)
payoffs_matrix = pd.DataFrame.from_dict(payoffs_matrix, orient="index")
print(payoffs_matrix)

# |%%--%%| <0wmVWPdvIj|VzyBhD5y8z>

population_size = 100
n_hawks = 1
population = [Strategy.HAWK] * n_hawks + [Strategy.DOVE] * (population_size - n_hawks)
assert len(population) == population_size
split = collections.Counter(population)

payoffs_matrix = simulate(population, n_fights=10000)
print(f"Population by species: {split}")

weighted_avg_payoffs = {}
for species, averaged_payoffs_by_species in payoffs_matrix.items():
    # 👇 For each species:
    # - Take the expected payoff per fight (against each other species)
    # - Weight it by the adversary's _frequency_ in the population
    # - Return the average value obtained.
    weighted_avg_payoffs[species] = mean(
        avg * split[s] / len(population)
        for s, avg in averaged_payoffs_by_species.items()
    )

# |%%--%%| <VzyBhD5y8z|iEYfgwpXY7>


# Make this a function to re-use it easily
def weighted_average_payoffs(payoffs_matrix, split) -> dict[Strategy, float]:
    w_avg_payoffs = {}
    for species, averaged_payoffs_by_species in payoffs_matrix.items():
        # 👇 For each species:
        # - Take the expected payoff per fight (against each other species)
        # - Weight it by the adversary's _frequency_ in the population
        # - Return the average value obtained.
        w_avg_payoffs[species] = mean(
            avg * split[s] / len(population)
            for s, avg in averaged_payoffs_by_species.items()
        )
    return w_avg_payoffs


# |%%--%%| <iEYfgwpXY7|m4FS1mQHcs>


population_size = 100
#         👇
n_hawks = 25
population = [Strategy.HAWK] * n_hawks + [Strategy.DOVE] * (population_size - n_hawks)
assert len(population) == population_size
split = collections.Counter(population)
print(f"Population by species: {split}")

payoffs_matrix = simulate(population, n_fights=10000)
w_avg_payoffs = weighted_average_payoffs(payoffs_matrix, split)
print(f"Weighted average payoff: {weighted_average_payoffs}")

#|%%--%%| <m4FS1mQHcs|NvwkI7SDMJ>


results = []
for n_hawks in range(population_size):
    population = [Strategy.HAWK] * n_hawks + [Strategy.DOVE] * (
        population_size - n_hawks
    )
    assert len(population) == population_size
    split = collections.Counter(population)

    # NOTE: This doesn't change, so we can re-use the one above.
    # payoffs_matrix = simulate(population, n_fights=10000)

    w_avg_payoffs = weighted_average_payoffs(payoffs_matrix, split)

    results.append(w_avg_payoffs)

f, ax = plt.subplots(figsize=(10, 4))
ax = pd.DataFrame(results).plot.line(ax=ax)
plt.xlabel("% of hawks in the population")
plt.ylabel("Weighted average payoff")
x = math.floor((7 / 12) * 100)
y = results[x][Strategy.HAWK]
ax.plot(x, y, 'b.')
plt.annotate("58.33% hawks", (x, y), (x + 1, y + 1))
plt.tight_layout()
plt.savefig("hawks_by_frequency.svg", format="svg")
