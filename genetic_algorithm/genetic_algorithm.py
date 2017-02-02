import random
import itertools
from operator import itemgetter
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math
#from multiprocessing import Pool
import time

class GeneticAlgorithm:
    def __init__(self, target, value_space, population_size=100, crossover_factor=0.85, mutation_probability=0.1, min_target=0.95, plot=False):
        self.target = target
        self.target_size = len(target)
        self.value_space = value_space
        self.population = []
        self.generation = 0
        self.population_size = population_size
        self.crossover_factor = crossover_factor
        self.mutation_probability = float(mutation_probability) / 100 * self.target_size
        self.min_target = min_target
        self.plot = plot
        self.num_crossovers = int(self.population_size * self.crossover_factor)
        self.select_boundary = self.population_size - self.num_crossovers
        self.crossover_indices = [i%self.select_boundary for i in range(self.num_crossovers)]
        self.indices = list(range(self.target_size))
        self.population_indices = list(range(self.population_size))
        random.seed(1)

    def crossover(self, index):
        child = []
        parent = random.choice(self.crossover_indices)
        for i in range(self.target_size):
            if self.population[parent][i] == self.population[index][i]:
                child.append(self.population[parent][i]) # values same as in the best individual stay
            else:
                child.append(random.choice(value_space)) # all other values are choosen randomly
        return self.mutate(child)

    def getFitness(self, genom):
        score = 0
        for i,val in enumerate(genom):
            if val == self.target[i]:
                score += 1
        return round((score / self.target_size),4) #* (score / self.target_size), 4) # nonlinear fitness function two better choose the best individuals

    def mutate(self, genom):
        # minimum one mutation
        i = random.choice(self.indices)
        genom[i] = random.choice(value_space) # change the value with a radomly selected one

        # compute order of mutation denpendent on the current fitness ()
        number_of_indices = int(self.mutation_probability / self.getFitness(self.population[0]))
        for j in self.indices[:number_of_indices]:
            i = random.choice(self.indices)
            genom[i] = random.choice(value_space) # change the value with a radomly selected one
        return genom

    def init_population(self):
        print("Initializing ...", end="")
        for j in range(self.population_size):
            individual = [random.choice(value_space) for i in np.arange(self.target_size)]
            self.population.append(individual)
        print(" Done!")

    def selection(self):
        fitness = [self.getFitness(i) for i in self.population]
        sorted_indices = list(zip(self.population_indices, fitness))
        sorted_indices.sort(key=itemgetter(1), reverse=True) # sorting just the indices may be quicker than the whole population
        self.population = [self.population[i[0]] for i in sorted_indices[:self.select_boundary]]

    def nextGeneration(self):
        #print("Population size before selection: ", len(self.population))
        self.selection()
        #print("Population size after selection: ", len(self.population))
        #print("Needed number of Crossovers: ", self.num_crossovers)
        #print("Crossover indices: ", self.crossover_indices)
        childs = [self.crossover(i) for i in self.crossover_indices]
        self.population += childs
        self.generation += 1

    def getBestIndividual(self):
        individual = ""
        for i in self.population[0]:
            individual += str(i)
        return individual

    def display(self):
        print ("")
        print ("Generation ", str(self.generation))
        position = 1
        for p in self.population[:5]:
            print ("Genom #", str(position), " has fitness of ", str(self.getFitness(p)))
            position += 1
        print ("")

    def step(self, update_threshold):
        last_fitness = self.getFitness(self.population[0])
        self.nextGeneration()
        while (self.getFitness(self.population[0]) < last_fitness + update_threshold):
            self.nextGeneration()
        return self.population[0]

    def hasConverged(self):
        if (self.getFitness(self.population[0]) > self.min_target):
            return True
        return False

    def getScore(self):
        return self.getFitness(self.population[0])

    def getSolution(self):
        self.init_population()
        while 1:
            current_fitness = self.getFitness(self.population[0])
            if (self.generation % 500 == 0):
                print ("Generation: ", self.generation, ", fitness: ", current_fitness)
            if (current_fitness > self.min_target):
                break
            self.nextGeneration()

        return self.population[0]

    def start(self):
        if self.plot:
            X = [i for i in range(self.target_size)]
            X2 = [i+.25 for i in range(self.target_size)]
        self.init_population()
        last_fitness = 0
        while 1:
            current_fitness = self.getFitness(self.population[0])
            if (current_fitness > self.min_target):
                break

            self.nextGeneration()

            if current_fitness > (last_fitness + 0.01):
                self.display()
                last_fitness = current_fitness
                if self.plot:
                    plt.cla()
                    plt.title(str(last_fitness))
                    plt.bar(X2, self.population[0], width=.5, color="red")
                    plt.plot(X, self.target, color="blue", linewidth=3)
                    plt.pause(.05)

        fitness = self.getFitness(self.population[0])

        print ("")
        print ("Finished!")
        print ("target found with fitness of ", str(fitness), " in Generation ", str(self.generation), ":")
        print (str(self.population[0]))
        print ("")
        print ("")

        if self.plot:
            plt.cla()
            plt.title(str(fitness))
            plt.bar(X, self.population[0], width=1, color="red")
            plt.plot(X, self.target, color="blue", linewidth=3)

            while True:
                plt.pause(0.05)


#################################################################
#################################################################

########## BINARY ##########
target_binary = np.array([
    0,0,0,1,1,1,0,0,1,1,1,1,0,0,0,1,1,0,0,0,0,0,0,1,1,
    1,1,1,0,0,0,1,1,0,0,0,0,1,1,1,0,0,1,1,1,1,1,1,0,0,
    0,0,0,1,1,1,0,0,1,1,1,1,0,0,0,1,1,0,0,0,0,0,0,1,1,
    1,1,1,0,0,0,1,1,0,0,0,0,1,1,1,0,0,1,1,1,1,1,1,0,0,
    0,0,0,1,1,1,0,0,1,1,1,1,0,0,0,1,1,0,0,0,0,0,0,1,1,
    1,1,1,0,0,0,1,1,0,0,0,0,1,1,1,0,0,1,1,1,1,1,1,0,0,
    0,0,0,1,1,1,0,0,1,1,1,1,0,0,0,1,1,0,0,0,0,0,0,1,1,
    1,1,1,0,0,0,1,1,0,0,0,0,1,1,1,0,0,1,1,1,1,1,1,0,0,
    0,0,0,1,1,1,0,0,1,1,1,1,0,0,0,1,1,0,0,0,0,0,0,1,1,
    1,1,1,0,0,0,1,1,0,0,0,0,1,1,1,0,0,1,1,1,1,1,1,0,0,
    0,0,0,1,1,1,0,0,1,1,1,1,0,0,0,1,1,0,0,0,0,0,0,1,1,
    0,0,1,1,1,0,0,1,1,1,1,0,0,0,1,1,0,0,0,0,0,0,1,1,0,
    1,1,1,0,0,0,1,1,0,0,0,0,1,1,1,0,0,1,1,1,1,1,1,0,0,
    0,0,0,1,1,1,0,0,1,1,1,1,0,0,0,1,1,0,0,0,0,0,0,1,1,
    1,1,1,0,0,0,1,1,0,0,0,0,1,1,1,0,0,1,1,1,1,1,1,0,0,
    0,0,0,1,1,1,0,0,1,1,1,1,0,0,0,1,1,0,0,0,0,0,0,1,1,
    1,1,1,0,0,0,1,1,0,0,0,0,1,1,1,0,0,1,1,1,1,1,1,0,0,
    0,0,0,1,1,1,0,0,1,1,1,1,0,0,0,1,1,0,0,0,0,0,0,1,1,
    1,1,1,0,0,0,1,1,0,0,0,0,1,1,1,0,0,1,1,1,1,1,1,0,0,
    0,0,0,1,1,1,0,0,1,1,1,1,0,0,0,1,1,0,0,0,0,0,0,1,1,
    1,1,1,0,0,0,1,1,0,0,0,0,1,1,1,0,0,1,1,1,1,1,1,0,0,
    0,0,0,1,1,1,0,0,1,1,1,1,0,0,0,1,1,0,0,0,0,0,0,1,1,
    1,1,1,0,0,0,1,1,0,0,0,0,1,1,1,0,0,1,1,1,1,1,1,0,0])
value_space_binary = [0,1]
##############################

########## DECIMAL ##########
target_decimal = np.array([
    0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,
    6,7,7,7,8,8,8,9,9,9,8,8,7,7,6,6,5,5,4,4,
    3,3,2,2,1,0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,
    6,7,7,7,8,8,8,9,9,9,8,8,7,7,6,6,5,5,4,4,
    3,3,2,2,1,5,5,5,6,6])
value_space_decimal = [0,1,2,3,4,5,6,7,8,9]
##############################

########## SINUS ##########
highest_number = 300

import math
def f(x):
    return int(highest_number*math.sin(math.pi*4*x/highest_number))

target_f = []
#for x in range(highest_number):
    #target_f.append(f(x))

##############################

########## PI DIGITS ##########
def make_pi():
    q, r, t, k, m, x = 1, 0, 1, 1, 3, 3
    for j in range(1000):
        if 4 * q + r - t < m * t:
            yield m
            q, r, t, k, m, x = 10*q, 10*(r-m*t), t, k, (10*(3*q+r))//t - 10*m, x
        else:
            q, r, t, k, m, x = q*k, (2*q+r)*x, t*x, k+1, (q*(7*k+2)+r*x)//(t*x), x+2

target_pi = []
for i in make_pi():
    target_pi.append(i)
value_space_pi = range(10) # [0,1,2,3,4,5,6,7,8,9]
##############################

########## MINIMUM IN FUNCTION ##########
target_minimum = [1,1,1,0,1,0,1,1,0,1,1,1,1,0,0,1,1,0,1,0,0,0,1,0,1,1,0,0,0,1] # == 500 in decimal

value_space_minimum = [0,1]
##############################

########## Shakespear ##########
target_shakespear = [
    "t","o"," ","b","e"," ","o","r"," ","n","o","t"," ","t",
    "o"," ","b","e"," ","t","h","a","t"," ","i","s"," ","t",
    "h","e"," ","q","u","e","s","t","i","o","n"] # to be or not to be that is the question

value_space_shakespear = [
    " ","a","b","c","d","e","f","g","h","i","j",
    "k","l","m","n","o","p","q","r","s","t","u",
    "v","w","x","y","z"]
##############################


if __name__ == '__main__':

    img = Image.open("circle.png") # open colour image
    img = img.convert('1') # convert image to black and white
    img.save("ground_truth.png")
    target_image = np.array(img.getdata())
    value_space_image = [0, 255] #range(256)

    print ("Starting generic algorithm ...")

    target = target_image
    value_space = value_space_image
    ga = GeneticAlgorithm(target, value_space)

    ga.start()
    #result = ga.getSolution()
    #x = np.reshape(result, (img.height, img.width))

    #mng = plt.get_current_fig_manager()
    #mng.resize(*mng.window.maxsize())

    start = time.time()

    ga.init_population()
    while not ga.hasConverged():
        if (ga.getScore() > 25):
            result = ga.step(10)
        else :
            result = ga.step(2.5)
        x = np.reshape(result, (img.height, img.width))
        time_elapsed = int(time.time() - start)
        plt.title("Generation: " + str(ga.generation) + ", fitness: " + str(ga.getScore()) + ", Time: " + str(time_elapsed) + "s")
        plt.imshow(x, cmap="gray")
        plt.pause(0.05)


    time_elapsed = int(time.time() - start)
    plt.title("Finished! "" + ""Generation: " + str(ga.generation) + ", fitness: " + str(ga.getScore()) + ", Time: " + str(time_elapsed) + "s")
    plt.imshow(x, cmap="gray")
    plt.show()
