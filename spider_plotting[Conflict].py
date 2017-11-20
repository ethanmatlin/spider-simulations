import numpy as np 
import random
import plotly
import plotly.plotly as py
import time
import plotly.graph_objs as go
import h5py

plotly.tools.set_credentials_file(username='ethanmatlin', api_key='3eIOZHrSqA7LoUdFIuGI')

num_spiders = 1000
# Actually number of locations is num_locations^2 since we're on a 2x2 grid.
num_locations = 10
xi = .1 
kappa = .1
T = 50
numMales=num_spiders

def number_at_location(row,col):
	num = 0
	#print(spider_locations)
	for i in range(len(spider_locations)):
		if (spider_locations[i][0]==row and spider_locations[i][1]==col):
			num = num + 1
	return num

# Randomly assign all 5 spiders a position
spider_locations = [(random.randint(0,num_locations-1), random.randint(0,num_locations-1)) for i in range(num_spiders)]
#print(spider_locations)
male_locations = [(random.randint(0,num_locations-1), random.randint(0,num_locations-1)) for i in range(numMales)]

#An array of vectors: each of the J spiders needs room for T different fitnesses over their lifetime.
F = [np.ones(T) for i in range(num_spiders)]

initial_number_at_locations = np.ones((num_locations, num_locations))
for k in range(num_locations):
	for l in range(num_locations):
		initial_number_at_locations[k, l] = number_at_location(k,l)

def size(F, j, t):
	sum = 1
	for tau in range(t):
		sum = sum + (3/t)*((4*(tau-t/2)**2)/t**2)*F[j][tau]
	return sum 

def dist(old_loc, new_loc):
	return ((old_loc[0]-new_loc[0])**2+(old_loc[1]-new_loc[1])**2)**.5

def fitness_func(newLoc, spider, F, t):
	if (spider_locations[spider] == (-99,-99)):
		return 0
	else:
		newLoc_row = newLoc[0] 
		newLoc_col = newLoc[1]
		return size(F, spider, t)/v(newLoc_row,newLoc_col, t, spider)*((u[newLoc_row][newLoc_col])) - xi*dist(spider_locations[spider],(newLoc_row,newLoc_col)) - kappa

def v(row,col, t, j):
	mass = 1
	for i in range(num_spiders):
		if (i!=j and spider_locations[i][0]==row and spider_locations[i][1]==col):
			mass = mass + F[i][t]
	return mass

def rargmax(vector):
    """ Argmax that chooses randomly among eligible maximum indices. """
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return random.choice(indices)


number_at_location_time_series = [np.ones((num_locations,num_locations)) for i in range(T)]
sizes_panel = [np.ones(T) for j in range(num_spiders)]
locations_panel = [[(np.ones(num_locations), np.ones(num_locations)) for t in range(T)] for j in range(num_spiders)]

#old code for locations panel locations_panel = [[(np.ones(num_locations), np.ones(num_locations)) for i in range(num_spiders)] for j in range(T)]


#print(number_at_location_time_series)

for t in range(T):
	print(t)
	u = [[np.random.poisson(5) for i in range(num_locations)] for k in range(num_locations)]
	#u = [[0 for i in range(num_locations)] for k in range(num_locations)]
	#u[1][1] = 100
	for j in range(num_spiders):
		# Spiders eat
		F[j][t] = fitness_func(spider_locations[j], j, F, t) 
		sizes_panel[j][t] = size(F, j, t)
		if (F[j][t] <= 0.0):
			spider_locations[j]=(-99,-99)
		else:
			# Spiders calculate fitness from other locations
			F_calc = [[fitness_func((i, k), j, F, t) for i in range(num_locations)] for k in range(num_locations)]
			F_calc = np.array(F_calc)
			i,j = np.unravel_index(rargmax(F_calc), F_calc.shape)
			spider_locations[j] = (i,j)
	
	matrix = np.ones((num_locations, num_locations))
	for k in range(num_locations):
		for l in range(num_locations):
			number_at_location_time_series[t][k,l] = number_at_location(k,l)

	locationsMat[t] = spider_locations

number_at_location_time_series[0]=initial_number_at_locations



print(number_at_location_time_series[0])
print(number_at_location_time_series[1])
print(number_at_location_time_series[2])
print(number_at_location_time_series[49])

with h5py.File('locationNum.h5', 'w') as hf:
    hf.create_dataset("number_at_location_time_series",  data=number_at_location_time_series)

with h5py.File('size.h5', 'w') as hf:
	hf.create_dataset("sizes_panel",  data=sizes_panel)

with h5py.File('locations.h5', 'w') as hf:
	hf.create_dataset("locations",  data=locationsMat)

# trace = go.Heatmap(z=number_at_location_time_series[0])
# data=[trace]
# py.iplot(data, filename='heatmap_1')

# trace = go.Heatmap(z=number_at_location_time_series[49])
# data=[trace]
# py.iplot(data, filename='heatmap_99')


# x = np.linspace(1,num_locations,num_locations)
# y = np.linspace(1,num_locations,num_locations)
# xx,yy = np.meshgrid(x,y)
# zz = number_at_location_time_series[0]

# trace = go.Heatmap(x=x,y=y,z=zz)
# frames = [ {
# 	'data': [{
# 	'z': number_at_location_time_series[n], 
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