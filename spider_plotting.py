import numpy as np 
import random
import plotly
import plotly.plotly as py
import time
import plotly.graph_objs as go
import h5py

plotly.tools.set_credentials_file(username='ethanmatlin', api_key='3eIOZHrSqA7LoUdFIuGI')

numSpiders = 1000
# Actually number of locations is numLocations^2 since we're on a 2x2 grid.
numLocations = 10
xi = .1 
kappa = .1
T = 50
numMales=numSpiders

def numAtLoc(loc_y,loc_x):
	counter = 0
	#print(spiderLocs)
	for i in range(len(spiderLocs)):
		if (spiderLocs[i][0]==loc_y and spiderLocs[i][1]==loc_x):
			counter = counter + 1
	return counter

# Randomly assign all 5 spiders a position
spiderLocs = [(random.randint(0,numLocations-1), random.randint(0,numLocations-1)) for i in range(numSpiders)]
#print(spiderLocs)
maleLocs = [(random.randint(0,numLocations-1), random.randint(0,numLocations-1)) for i in range(numMales)]

#An array of arrays: each of the J spiders needs room for T different fitnesses over their lifetime.
F = [np.ones(T) for i in range(numSpiders)]

start_matrix = np.ones((numLocations, numLocations))
for k in range(numLocations):
	for l in range(numLocations):
		start_matrix[k, l] = numAtLoc(k,l)

def size(F, j, t):
	sum = 1
	for tau in range(t):
		sum = sum + (3/t)*((4*(tau-t/2)**2)/t**2)*F[j][tau]
	return sum 

def dist(old_loc, new_loc):
	return ((old_loc[0]-new_loc[0])**2+(old_loc[1]-new_loc[1])**2)**.5

def fitness_func(newLoc, spider, F, t):
	if (spiderLocs[spider] == (-99,-99)):
		return 0
	else:
		newLoc_y = newLoc[0] 
		newLoc_x = newLoc[1]
		return size(F, spider, t)/v(newLoc_x,newLoc_y, t, spider)*((u[newLoc_y][newLoc_x])) - xi*dist(spiderLocs[spider],(newLoc_y,newLoc_x)) - kappa
		#return F[spider][t]/(v(newLoc_x,newLoc_y, t, spider)+F[spider][t])*((u[newLoc_x][newLoc_y])) - xi*dist(spiderLocs[spider],(newLoc_x,newLoc_y)) - kappa

def v(loc_x,loc_y, t, j):
	mass = 1
	for i in range(numSpiders):
		if (i!=j and spiderLocs[i][0]==loc_y and spiderLocs[i][1]==loc_y):
			mass = mass + F[i][t]
	return mass

def rargmax(vector):
    """ Argmax that chooses randomly among eligible maximum indices. """
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return random.choice(indices)



allMats = [np.ones((numLocations,numLocations)) for i in range(T)]
sizeMat = [np.ones(T) for j in range(numSpiders)]
locationsMat = [[(np.ones(numLocations), np.ones(numLocations)) for i in range(numSpiders)] for j in range(T)]


#print(allMats)

for t in range(T):
	print(t)
	u = [[np.random.poisson(5) for i in range(numLocations)] for k in range(numLocations)]
	#u = [[0 for i in range(numLocations)] for k in range(numLocations)]
	#u[1][1] = 100
	for j in range(numSpiders):
		# Spiders eat
		F[j][t] = fitness_func(spiderLocs[j], j, F, t) 
		sizeMat[j][t] = size(F, j, t)
		if (F[j][t] <= 0.0):
			spiderLocs[j]=(-99,-99)
		else:
			# Spiders calculate fitness from other locations
			F_calc = [[fitness_func((i, k), j, F, t) for i in range(numLocations)] for k in range(numLocations)]
			F_calc = np.array(F_calc)
			i,j = np.unravel_index(rargmax(F_calc), F_calc.shape)
			spiderLocs[j] = (i,j)
	
	matrix = np.ones((numLocations, numLocations))
	for k in range(numLocations):
		for l in range(numLocations):
			matrix[k, l] = numAtLoc(k,l)
	allMats[t] = matrix
	locationsMat[t] = spiderLocs
allMats[0]=start_matrix
print(allMats[0])
print(allMats[1])
print(allMats[2])
print(allMats[49])

with h5py.File('locationNum.h5', 'w') as hf:
    hf.create_dataset("allMats",  data=allMats)

with h5py.File('size.h5', 'w') as hf:
	hf.create_dataset("sizeMat",  data=sizeMat)

with h5py.File('locations.h5', 'w') as hf:
	hf.create_dataset("locations",  data=locationsMat)

trace = go.Heatmap(z=allMats[0])
data=[trace]
py.iplot(data, filename='heatmap_1')

trace = go.Heatmap(z=allMats[49])
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