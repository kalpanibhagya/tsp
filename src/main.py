# encoding:utf-8

'''
	Solution for Travelling Salesman Problem using PSO (Particle Swarm Optimization)
	Discrete PSO for TSP

	References: 
		http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.258.7026&rep=rep1&type=pdf
		http://www.cs.mun.ca/~tinayu/Teaching_files/cs4752/Lecture19_new.pdf
		http://www.swarmintelligence.org/tutorials.php

	References are in the folder "references" of the repository.
'''

from operator import attrgetter
import random, sys, time, copy

from sys import argv

import numpy as np
import pandas as pd
import math

from io_helper import read_tsp, normalize

from distance import select_closest, euclidean_distance, route_distance

# class that represents a graph
class Graph:

	def __init__(self, amount_vertices):
		self.edges = {} # dictionary of edges
		self.vertices = set() # set of vertices
		self.amount_vertices = amount_vertices # amount of vertices


	# adds a edge linking "src" in "dest" with a "cost"
	def addEdge(self, src, dest, cost = 0):
		# checks if the edge already exists
		if not self.existsEdge(src, dest):
			self.edges[(src, dest)] = cost
			self.vertices.add(src)
			self.vertices.add(dest)


	# checks if exists a edge linking "src" in "dest"
	def existsEdge(self, src, dest):
		return (True if (src, dest) in self.edges else False)


	# shows all the links of the graph
	def showGraph(self):
		print('Showing the graph:\n')
		for edge in self.edges:
			print('%d linked in %d with cost %d' % (edge[0], edge[1], self.edges[edge]))

	# returns total cost of the path
	def getCostPath(self, path):
		
		total_cost = 0
		for i in range(self.amount_vertices - 1):
			total_cost += self.edges[(path[i], path[i+1])]

		# add cost of the last edge
		total_cost += self.edges[(path[self.amount_vertices - 1], path[0])]
		return total_cost


	# gets random unique paths - returns a list of lists of paths
	def getRandomPaths(self, max_size):

		random_paths, list_vertices = [], list(self.vertices)

		initial_vertice = random.choice(list_vertices)
		if initial_vertice not in list_vertices:
			print('Error: initial vertice %d not exists!' % initial_vertice)
			sys.exit(1)

		list_vertices.remove(initial_vertice)
		list_vertices.insert(0, initial_vertice)

		for i in range(max_size):
			list_temp = list_vertices[1:]
			random.shuffle(list_temp)
			list_temp.insert(0, initial_vertice)

			if list_temp not in random_paths:
				random_paths.append(list_temp)

		return random_paths


# class that represents a complete graph
class CompleteGraph(Graph):

	# generates a complete graph
	def generates(self):
		for i in range(self.amount_vertices):
			for j in range(self.amount_vertices):
				if i != j:
					weight = random.randint(1, 10)
					self.addEdge(i, j, weight)


# class that represents a particle
class Particle:

	def __init__(self, solution, cost):

		# current solution
		self.solution = solution

		# best solution (fitness) it has achieved so far
		self.pbest = solution

		# set costs
		self.cost_current_solution = cost
		self.cost_pbest_solution = cost

		# velocity of a particle is a sequence of 4-tuple
		# (1, 2, 1, 'beta') means SO(1,2), prabability 1 and compares with "beta"
		self.velocity = []

	# set pbest
	def setPBest(self, new_pbest):
		self.pbest = new_pbest

	# returns the pbest
	def getPBest(self):
		return self.pbest

	# set the new velocity (sequence of swap operators)
	def setVelocity(self, new_velocity):
		self.velocity = new_velocity

	# returns the velocity (sequence of swap operators)
	def getVelocity(self):
		return self.velocity

	# set solution
	def setCurrentSolution(self, solution):
		self.solution = solution

	# gets solution
	def getCurrentSolution(self):
		return self.solution

	# set cost pbest solution
	def setCostPBest(self, cost):
		self.cost_pbest_solution = cost

	# gets cost pbest solution
	def getCostPBest(self):
		return self.cost_pbest_solution

	# set cost current solution
	def setCostCurrentSolution(self, cost):
		self.cost_current_solution = cost

	# gets cost current solution
	def getCostCurrentSolution(self):
		return self.cost_current_solution

	# removes all elements of the list velocity
	def clearVelocity(self):
		del self.velocity[:]


# PSO algorithm
class PSO:

	def __init__(self, graph, iterations, size_population, beta=1, alfa=1):
		self.graph = graph # the graph
		self.iterations = iterations # max of iterations
		self.size_population = size_population # size population
		self.particles = [] # list of particles
		self.beta = beta # the probability that all swap operators in swap sequence (gbest - x(t-1))
		self.alfa = alfa # the probability that all swap operators in swap sequence (pbest - x(t-1))

		# initialized with a group of random particles (solutions)
		solutions = self.graph.getRandomPaths(self.size_population)

		# checks if exists any solution
		if not solutions:
			print('Initial population empty! Try run the algorithm again...')
			sys.exit(1)

		# creates the particles and initialization of swap sequences in all the particles
		for solution in solutions:
			# creates a new particle
			particle = Particle(solution=solution, cost=graph.getCostPath(solution))
			# add the particle
			self.particles.append(particle)

		# updates "size_population"
		self.size_population = len(self.particles)


	# set gbest (best particle of the population)
	def setGBest(self, new_gbest):
		self.gbest = new_gbest

	# returns gbest (best particle of the population)
	def getGBest(self):
		return self.gbest


	# shows the info of the particles
	def showsParticles(self):

		print('Showing particles...\n')
		for particle in self.particles:
			print('pbest: %s\t|\tcost pbest: %d\t|\tcurrent solution: %s\t|\tcost current solution: %d' \
				% (str(particle.getPBest()), particle.getCostPBest(), str(particle.getCurrentSolution()),
							particle.getCostCurrentSolution()))
		print('')


	def run(self):

		# for each time step (iteration)
		for t in range(self.iterations):

			# updates gbest (best particle of the population)
			self.gbest = min(self.particles, key=attrgetter('cost_pbest_solution'))

			# for each particle in the swarm
			for particle in self.particles:

				particle.clearVelocity() # cleans the speed of the particle
				temp_velocity = []
				solution_gbest = copy.copy(self.gbest.getPBest()) # gets solution of the gbest
				solution_pbest = particle.getPBest()[:] # copy of the pbest solution
				solution_particle = particle.getCurrentSolution()[:] # gets copy of the current solution of the particle

				# generates all swap operators to calculate (pbest - x(t-1))
				for i in range(self.graph.amount_vertices):
					if solution_particle[i] != solution_pbest[i]:
						# generates swap operator
						swap_operator = (i, solution_pbest.index(solution_particle[i]), self.alfa)

						# append swap operator in the list of velocity
						temp_velocity.append(swap_operator)

						# makes the swap
						aux = solution_pbest[swap_operator[0]]
						solution_pbest[swap_operator[0]] = solution_pbest[swap_operator[1]]
						solution_pbest[swap_operator[1]] = aux

				# generates all swap operators to calculate (gbest - x(t-1))
				for i in range(self.graph.amount_vertices):
					if solution_particle[i] != solution_gbest[i]:
						# generates swap operator
						swap_operator = (i, solution_gbest.index(solution_particle[i]), self.beta)

						# append swap operator in the list of velocity
						temp_velocity.append(swap_operator)

						# makes the swap
						aux = solution_gbest[swap_operator[0]]
						solution_gbest[swap_operator[0]] = solution_gbest[swap_operator[1]]
						solution_gbest[swap_operator[1]] = aux

				
				# updates velocity
				particle.setVelocity(temp_velocity)

				# generates new solution for particle
				for swap_operator in temp_velocity:
					if random.random() <= swap_operator[2]:
						# makes the swap
						aux = solution_particle[swap_operator[0]]
						solution_particle[swap_operator[0]] = solution_particle[swap_operator[1]]
						solution_particle[swap_operator[1]] = aux
				
				# updates the current solution
				particle.setCurrentSolution(solution_particle)
				# gets cost of the current solution
				cost_current_solution = self.graph.getCostPath(solution_particle)
				# updates the cost of the current solution
				particle.setCostCurrentSolution(cost_current_solution)

				# checks if current solution is pbest solution
				if cost_current_solution < particle.getCostPBest():
					particle.setPBest(solution_particle)
					particle.setCostPBest(cost_current_solution)
		

if __name__ == "__main__":
	
	# creates the Graph instance
	graph = Graph(amount_vertices=5)

	problem = read_tsp(argv[1])
	#print(type(problem))

	#distance = route_distance(problem)
	#print(distance)
	#print(distance.itemsize)
	
	#print(problem['x'].iloc[0])

	# using apply function to create a new column 
	#problem['cost'] = problem.apply(lambda row: row.x - row.y, axis = 1) 
	
	points = problem[['city', 'x', 'y']]
	#points2 = pd.DataFrame(np.roll(points, 1, axis=0))
	#print(type(points2.columns.map(type)))
	#points2.columns = ['city2','x2', 'y2']
	#print(points2["0"])
	#points2 = points2.rename(columns={"0":'x1', "1":'y1'}, inplace=True)
	#print(points2)
	#print(points2['x2'].iloc[0])
	# Print the DataFrame after addition 
	# of new column 
	#print(problem)
	#print(len(points.index))
	columns = ['City1','City2', 'Cost']
	df_new = pd.DataFrame(columns=columns)

	items = range(1,5)
	for i in items:
		points2 = pd.DataFrame(np.roll(points, i, axis=0))
		points2.columns = ['city2','x2', 'y2']
		for j in range(0,len(points.index)):
			xcost = points['x'].iloc[j] - points2['x2'].iloc[j]
			ycost = points['y'].iloc[j] - points2['y2'].iloc[j]
			
			distance = math.sqrt(xcost**2 + ycost**2)

			x = round(distance)
			df_new = df_new.append({'City1' : int(points['city'].iloc[j]) , 'City2' : int(points2['city2'].iloc[j]), 'Cost' : x} , ignore_index=True)
			#print(x)
	

	print(df_new)
	listnew = df_new.values.tolist()
	print(listnew)
	#y = np.unique(listnew)
	tspGraph = set(tuple(x) for x in listnew)
	print(tspGraph)
	'''
	mySet = set((x,y) for x in range(1,6) for y in range(1,6))
	print(mySet)
	sortedSet = sorted(mySet, key=lambda element: (element[0], element[1]))
	print(sortedSet)
	'''
	# This graph is in the folder "images" of the repository.

	#print(type(tspGraph))
	
	for value1, value2, key in tspGraph:
		graph.addEdge(value1, value2, key)

	# creates a PSO instance 

	pso = PSO(graph, iterations=100, size_population=10, beta=1, alfa=0.9)
	pso.run() # runs the PSO algorithm
	pso.showsParticles() # shows the particles

	# shows the global best particle
	print('gbest: %s | cost: %d\n' % (pso.getGBest().getPBest(), pso.getGBest().getCostPBest()))

	'''
	# random graph
	print('Random graph...')
	random_graph = CompleteGraph(amount_vertices=20)
	random_graph.generates()
	pso_random_graph = PSO(random_graph, iterations=10000, size_population=10, beta=1, alfa=1)
	pso_random_graph.run()
	print('gbest: %s | cost: %d\n' % (pso_random_graph.getGBest().getPBest(), 
					pso_random_graph.getGBest().getCostPBest()))
	'''

'''
def main():
    if len(argv) != 2:
        print("Correct use: python src/main.py <filename>.tsp")
        return -1

    problem = read_tsp(argv[1])
    print('problem set')
    print(problem)

    distance = route_distance(problem)

    print('Route found of length {}'.format(distance))

    


if __name__ == '__main__':
    main()
'''
