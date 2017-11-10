import numpy as np 
import random
import plotly
import plotly.plotly as py
import time
import plotly.graph_objs as go

plotly.tools.set_credentials_file(username='ethanmatlin', api_key='3eIOZHrSqA7LoUdFIuGI')



numSpiders = 100
# Actually number of locations is numLocations^2 since we're on a 2x2 grid.
numLocations = 5
xi = 1 
kappa = 1
T = 100
numMales=numSpiders

# Randomly assign all 5 spiders a position
spiderLocs = [(random.randint(0,numLocations-1), random.randint(0,numLocations-1)) for i in range(numSpiders)]
print(spiderLocs)
maleLocs = [(random.randint(0,numLocations-1), random.randint(0,numLocations-1)) for i in range(numMales)]

#An array of arrays: each of the J spiders needs room for T different fitnesses over their lifetime.
F = [np.ones(T) for i in range(numSpiders)]



def size(F, j, t):
	sum = 1
	for tau in range(t):
		sum = sum + ((4*(tau-t/2)**2)/t**2)*F[j][tau]
	return sum 

def dist(old_loc, new_loc):
	return 1

def fitness_func(newLoc, spider, F, t):
	newLoc_x = newLoc[0] 
	newLoc_y = newLoc[1]
	#print(newLoc_y)
	#print(newLoc_x)
	#aaa
	return size(F, spider, t)/v(newLoc_x,newLoc_y, t)*((u[newLoc_x][newLoc_y])) - xi*dist(spiderLocs[spider],(newLoc_x,newLoc_y)) - kappa

def v(loc_x,loc_y, t):
	mass = 1 
	for i in range(numSpiders):
		if (spiderLocs[i][0]==loc_x & spiderLocs[i][1]==loc_y):
			mass = mass + F[i][t]
	return mass

def rargmax(vector):
    """ Argmax that chooses randomly among eligible maximum indices. """
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return random.choice(indices)

def numAtLoc(loc_x,loc_y):
	counter = 0
	for i in range(len(spiderLocs)):
		if (spiderLocs[i][0]==loc_x & spiderLocs[i][1]==loc_y):
			counter = counter + 1
	return counter

allMats = [np.ones((numLocations,numLocations)) for i in range(100)]
print(allMats)

for t in range(T):
	u = [[np.random.poisson(2) for i in range(numLocations)] for k in range(numLocations)]
	#print(u)
	for j in range(numSpiders):
		# Spiders eat
		F[j][t] = fitness_func(spiderLocs[j], j, F, t) 
		# Spiders calculate fitness from other locations
		F_calc = [[fitness_func((i, k), j, F, t) for i in range(numLocations)] for k in range(numLocations)]
		#print(F_calc)
		#aaa
		#randomize this in case there's a tie!
		F_calc = np.array(F_calc)
		#print(F_calc)
		i,j = np.unravel_index(np.argmax(F_calc), F_calc.shape)
		#i,j = np.unravel_index(rargmax(F_calc), F_calc.shape)
		#print(i)
		#print(j)
		spiderLocs[j] = (i,j)
	print(spiderLocs)
	
	matrix = np.ones((numLocations, numLocations))
	for k in range(numLocations):
		for l in range(numLocations):
			matrix[k, l] = numAtLoc(k,l)
	print(matrix)
	allMats[t] = matrix
	#print(i,j)
print(allMats)

trace = go.Heatmap(z=allMats[0])
data=[trace]
py.iplot(data, filename='heatmap_1')

trace = go.Heatmap(z=allMats[99])
data=[trace]
py.iplot(data, filename='heatmap_99')

# x = np.linspace(1,numLocations,numLocations)
# y = np.linspace(1,numLocations,numLocations)
# xx,yy = np.meshgrid(x,y)
# zz = allMats[0]

# trace = go.Heatmap(x=x,y=y,z=zz)
# frames = [ {
# 	'data': [{
# 	'z': allMats[n], 
# 	'type': 'heatmap'
# 	}]
# 	} for n in range(T)
# ]

# data = [trace]
# fig = go.Figure(data=data, frames= frames)
# py.icreate_animations(fig, filename='heatmap_anim'+str(time.time()))          
          
#fig=dict(data=data, layout=layout, frames=frames)  
#py.icreate_animations(fig, filename='animheatmap'+str(time.time()))

#ways to test. what is answer should be
# pcolor in matlab
#-if all prey is at one spot, spiders should all go there
#-etc.
# plot fitness function over time and see if agrees with what should be happening
# plot fitness function over space 