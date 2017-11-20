import numpy as np 
import random
import plotly
import plotly.plotly as py
import time
import plotly.graph_objs as go
import h5py

plotly.tools.set_credentials_file(username='ethanmatlin', api_key='3eIOZHrSqA7LoUdFIuGI')

num_spiders = 100
# Actually number of locations is num_locations^2 since we're on a 2x2 grid.
num_locations = 5
xi = 2
kappa = 1
T = 50
numMales=num_spiders


def number_at_location(row,col, t, location):
	num = 0
	#print(spider_locations)
	for j in range(num_spiders):
		if (location[j,0]==row and location[j,1]==col and location[j,0]>=0):
			num = num + 1
	return num

locations_panel = [[(10000,100000) for t in range(T)] for j in range(num_spiders)]
# Randomly assign all spiders a position
for j in range(num_spiders):
	locations_panel[j][0] = (random.randint(0,num_locations-1), random.randint(0,num_locations-1))

locations_panel = np.array(locations_panel)

#print(spider_locations)
#male_locations = [(random.randint(0,num_locations-1), random.randint(0,num_locations-1)) for i in range(numMales)]

#An array of vectors: each of the J spiders needs room for T different fitnesses over their lifetime.
F = [np.ones(T) for i in range(num_spiders)]

initial_number_at_locations = np.ones((num_locations, num_locations))

for row in range(num_locations):
	for col in range(num_locations):
		initial_number_at_locations[row, col] = number_at_location(row,col,0, locations_panel[:,0])


def size(F, j, t):
	sum = 1
	for tau in range(t):
		sum = sum + (3/t)*((4*(t/2-tau)**2)/t**2)*F[j][tau]
	return sum 

def dist(old_loc, new_loc):
	return ((old_loc[0]-new_loc[0])**2+(old_loc[1]-new_loc[1])**2)**.5

def fitness_func(newLoc, spider, F, t, location, locations_of_others):
	if (location[0]<0):
		return -20
	else:
		newLoc_row = newLoc[0] 
		newLoc_col = newLoc[1]
		# if (dist(locations_panel[spider][t],newLoc)!=0):
		# 	print("a")
		# 	print(size(F, spider, t))
		# 	print(v(newLoc_row,newLoc_col, t, spider))
		# 	print(u[newLoc_row][newLoc_col])
		# 	print("a")
		return size(F, spider, t)/v(newLoc_row,newLoc_col, t, spider, locations_of_others)*((u[newLoc_row][newLoc_col])) - xi*dist(location,newLoc) - kappa

def v(row,col, t, k, location):
	mass = 1
	for j in range(num_spiders):
		if (j!=k and location[j,0]==row and location[j,1]==col and F[j][t]>=0):
			mass = mass + F[j][t]
	return mass

def rargmax(vector):
    """ Argmax that chooses randomly among eligible maximum indices. """
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return random.choice(indices)


number_at_location_time_series = [np.ones((num_locations,num_locations)) for i in range(T)]
number_at_location_time_series[0] = initial_number_at_locations

sizes_panel = [np.ones(T) for j in range(num_spiders)]
#locationsMat = [[(np.ones(num_locations), np.ones(num_locations)) for i in range(num_spiders)] for t in range(T)]

#print(number_at_location_time_series)

for t in range(T):
	u = [[np.random.poisson(5) for i in range(num_locations)] for k in range(num_locations)]
	#u = [[0 for i in range(num_locations)] for k in range(num_locations)]
	#u[1][1] = 100
	print(t)
	for j in range(num_spiders):
		if (locations_panel[j][t][0]>=0):
			# Spiders eat
			F[j][t] = fitness_func(locations_panel[j,t], j, F, t, locations_panel[j,t],locations_panel[:,t])
			sizes_panel[j][t] = size(F, j, t)
		else:
			F[j][t] = -20 
			sizes_panel[j][t] = 0

		#if (j==0):
			# print("a")
			# print(locations_panel[j,t])
			# print("b")
		# if ((t+1)<(T)):
		# 	locations_panel[j][t+1]=(3,3)
		#print(u)
		if (sizes_panel[j][t] < 0.0):
			if ((t+1)<(T)):
				locations_panel[j,t+1, 0]=-99
				locations_panel[j,t+1, 1]=-99
		else:
			# Spiders calculate fitness from other locations
			F_calc = [[fitness_func((i, k), j, F, t, locations_panel[j,t],locations_panel[:,t]) for i in range(num_locations)] for k in range(num_locations)]
			F_calc = np.array(F_calc)
			#print(F_calc)
			k,l = np.unravel_index(np.argmax(F_calc), F_calc.shape)
			#print(k,l)
			#wait = input("PRESS ENTER TO CONTINUE.")
			if ((t+1)<(T)):
				t_ahead = t+1
				locations_panel[j,t_ahead, 0] = k
				locations_panel[j,t_ahead, 1] = l
				#if (k==0):
					#print(locations_panel[j][t+1])

	for row in range(num_locations):
		for col in range(num_locations):
			number_at_location_time_series[t][row,col] = number_at_location(row,col, t, locations_panel[:,t])
	#print(number_at_location_time_series[t])
	#print("a")
	#print(sizes_panel[1][t])
	#print(locations_panel[1][t])
			
	#locationsMat[t] = spider_locations

#locations_panel
#for t in len(locationsMat):



number_at_location_time_series[0]=initial_number_at_locations


for t in range(T):
	print(t)
	print(number_at_location_time_series[t])
	print(locations_panel[1,t,0])
	print(sizes_panel[1][t])


with h5py.File('locationNum.h5', 'w') as hf:
    hf.create_dataset("number_at_location_time_series",  data=number_at_location_time_series)

with h5py.File('size.h5', 'w') as hf:
	hf.create_dataset("sizes_panel",  data=sizes_panel)

with h5py.File('locations.h5', 'w') as hf:
	hf.create_dataset("locations_panel",  data=locations_panel)

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