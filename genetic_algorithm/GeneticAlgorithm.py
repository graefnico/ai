"""

Copyright (c) 2017 Nico Gräf (www.nicograef.de)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import random
from operator import itemgetter

class GeneticAlgorithm:
    def __init__(self, individual_size, fitness_function, solution_space=[0,1], population_size=100, crossover_factor=0.85, mutation_probability=0.01):

        # how many genes does an individual have
        self.individual_size = individual_size

        # the fitness function to evaluate individuals performance
        self.calcFitness = fitness_function

        # value space for the individuals
        self.solution_space = solution_space

        # creating a list for the population with minimum of 10 individuals
        # individuals are created in the init_population() method
        self.population_size = max(10, population_size)
        self.population = []

        # later we have to iterate often over the population,
        # so it's better to store the indices in a list
        self.population_indices = list(range(self.population_size))

        # starting with generation 0
        self.generation = 0

        # the probability that a gene/ gets mutated
        self.mutation_probability = mutation_probability

        # how many individuals are discarted in the natural selection of every generation
        self.num_crossovers = int(self.population_size * crossover_factor)

        # how many of the individuals are "copied" to the next generation
        self.select_boundary = self.population_size - self.num_crossovers

        # after sorting the individuals in every generation by their fitness,
        # those indices were "copied" to the next generation, starting from best fitness
        # Then the individuals at those indices are crossed to create new individuals.
        # Modulo needed to avoid and index (out of bounds) error.
        self.crossover_indices = [i % self.select_boundary for i in range(self.num_crossovers)]

        # the indices of the individual's properties (genes)
        # often used, so it's better to compute and store
        self.individual_indices = list(range(self.individual_size))

        # seed random for comparison
        random.seed(1)

        # variable to store current generation count
        self.generation = 0

        # create first generation by randomly creating individuals
        self.init_population()


    def init_population(self):
        """ Initializes the population with individuals of random genes """
        for _ in self.population_indices:
            individual = [random.choice(self.solution_space) for _ in self.individual_indices]
            self.population.append(individual)

    def selection(self):
        """
        The natural selection of the population.
        Sorts the population by the fitness of the individuals.
        Then discards the last [self.num_crossovers] individuals from the population.
        """
        fitness = [self.calcFitness(i) for i in self.population]
        sorted_indices = list(zip(self.population_indices, fitness))
        sorted_indices.sort(key=itemgetter(1), reverse=True) # sorting just the indices may be quicker than the whole population
        self.population = [self.population[i[0]] for i in sorted_indices[:self.select_boundary]]

    def crossover(self, parent1, parent2):
        """
        Creates a new individual (the child) by crossing two individuals (the parents).
        Those genes that have the same value in both parents are "copied" to the child.
        All other genes get random values.
        child[i] = (parent1[i] if parent1[i]==parent2[i], random.choice(solution_space) otherwise)
        """
        child = []
        for i in self.individual_indices:
            if self.population[parent1][i] == self.population[parent2][i]:
                child.append(self.population[parent1][i]) # values same as in the best individual stay
            else:
                child.append(random.choice(self.solution_space)) # all other values are choosen randomly
        return self.mutate(child)

    def mutate(self, individual):
        """
        Mutates every gene in the given individual with a probability of [self.mutation_probability]
        Mutation means replacing the current value with a random value from the solution space [self.solution_space].
        """"
        for i in self.individual_indices:
            if random.random() <= self.mutation_probability:
                individual[i] = random.choice(self.solution_space)
        return individual

    def nextGeneration(self):
        """
        Performs the (natural) selection of the current population.
        Then extend the population by crossing the best individuals,
        such that the population size stays the same.
        """
        self.selection()
        for i in self.crossover_indices:
            self.population.append(self.crossover(i, random.choice(self.crossover_indices))
        self.generation += 1

    def getBestIndividual(self):
        """ Returns the best individual so far, which is always at index 1 because of sorting before selection. """
        return self.population[0]

    def getCurrentFitness(self):
        """ Returns the current max fitness (= fitness of the best individual). """
        return self.calcFitness(self.getBestIndividual())

    def printCurrentFitness(self):
        """ Prints the current max fitness. """
        print("Current fitness is", self.getCurrentFitness())

    def solutionToString(self):
        """ Returns the best individual as a string. """
        solution = ""
        for i in self.getBestIndividual():
            solution += str(i)
        return solution
