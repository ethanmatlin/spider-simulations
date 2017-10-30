import numpy as np 
import random


numSpiders = 5
# Actually number of locations is numLocations^2 since we're on a 2x2 grid.
numLocations = 5
xi = 1 
kappa = 1
T = 100

# Randomly assign all 5 spiders a position
spiderLocs = [(random.randint(0,numLocations-1), random.randint(0,numLocations-1)) for i in range(numSpiders)]
#An array of arrays: each of the J spiders needs room for T different fitnesses over their lifetime.
F = [np.ones(T) for i in range(numSpiders)]

u = [[np.random.poisson(1) for i in range(numLocations)] for k in range(numLocations)]

def size(F, j, t):
	sum = 0 
	for tau in range(t):
		sum = sum + ((4*(tau-t/2)**2)/t**2)*F[j][tau]
	return sum 

def dist(old_loc, new_loc):
	return 1

def fitness_func(newLoc_x, newLoc_y, spider, F, t):
	return size(F, spider, t)*((u[newLoc_x][newLoc_y]/v(newLoc_x,newLoc_y))+v(newLoc_x,newLoc_y)) - xi*dist(spiderLocs[spider],(newLoc_x,newLoc_y)) - kappa

def v(loc_x,loc_y):
	counter = 0 
	for i in range(len(spiderLocs)):
		if (spiderLocs[i][0]==loc_x & spiderLocs[i][1]==loc_y):
			counter = counter + 1
	if (counter==0):
		counter=0.0001
	return counter


for t in range(T):
	for j in range(numSpiders):
		# Spiders eat
		#s(j) = 
		# Spiders calculate fitness from other locations
		F_calc = [[fitness_func(i, k, j, F, t) for i in range(numLocations)] for k in range(numLocations)]
		print(np.argmin(F_calc))
