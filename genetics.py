import numpy as np
from random import randint
import copy

class GAKill(Exception):
	def __init__(self, message):
		self.message = message

class Gene:
	fitness = 0
	score = 0
	genotype = []
	cursor = 0

	def __init__(self):
		pass

	def encode(self):
		pass

	def decode(self):
		pass

	def evaluate(self):
		pass

	def mutate(self, rate):
		gen_len = len(self.genotype)

		# Izbor nasumicnog hromozoma
		idx = np.random.random_integers(0, gen_len-1, size=(1, int(round(rate*gen_len))))

		# Mutiranje hromozoma
		self.genotype[idx] += 0.1 * (2 * np.random.random_sample(1) - 1)

	def read_genotype(self, delta):
		chunk = self.genotype[self.cursor:self.cursor + delta]
		self.cursor += delta
		return chunk


class GeneticAlgorithm:
	popsize = 0
	error = 1
	epoch = 0
	armageddon = 0

	def __init__(self, epochs, mutation_rate, data, targets, obj, args):
                #Konstruktor
		self.obj = obj
		self.args = args
		self.mutation_rate = mutation_rate
		self.training_data = data
		self.targets = targets
		self.armageddon = epochs

	def populate(self, size):
		# Konstruktor za kreiranje populacije
		self.population = [self.obj(self.args) for _ in range(size)]
		self.popsize = size

	def singleton(self):
		return self.obj(self.args, build=False)

	def evaluate(self):
		for gene in self.population:
			gene.evaluate(self.training_data, self.targets)

		self.population = sorted(self.population, key=lambda gene: gene.fitness)
		self.error = 1 - self.fittest().fitness  # Postavljanje opste greske

	def crossover(self):
		# Kreiranje nove populacije koriscenjem ruletske selekcije
		population = [self.breed(self.roulette(2)) for _ in range(self.popsize)]
		self.population = population

	def breed(self, parents):
		# Pravljenje novog gena
		offspring = self.singleton()

		
		length = parents[0].genotype.size - 1
		cuts = [randint(0, round(length/2)), randint(round(length/2), length)]

		
		offspring.genotype = np.concatenate((parents[0].genotype[:cuts[0]],
		parents[1].genotype[cuts[0]:cuts[1]], parents[0].genotype[cuts[1]:]))

		offspring.mutate(self.mutation_rate)
		offspring.decode()

		return offspring

	def roulette(self, n):
		choice = self.population[-self.popsize/2:]

		# Izracunavanje fitnes funkcije
		fitnesses = map(lambda x: x.fitness, choice)
		fitnesses /= np.sum(fitnesses) # Normalise

		return np.random.choice(choice, n, p=fitnesses)

	def fittest(self):
		# Kloniranje gena sa najboljim koeficijentom
		return copy.deepcopy(self.population[-1])

	def evolve(self):
		return True if self.epoch < self.armageddon else False
