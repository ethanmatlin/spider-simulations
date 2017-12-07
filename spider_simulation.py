import numpy as np 
import random
import plotly
import plotly.plotly as py
import time
import plotly.graph_objs as go
import h5py

plotly.tools.set_credentials_file(username='ethanmatlin', api_key='3eIOZHrSqA7LoUdFIuGI')

"""This script simulates spider movements using a agent-based discrete choice model. Spiders gain size (which is an accumulation of their fitness at 
	each time step) over time. If their size drops below zero, they die (meaning they aren't included in the model after that point). Spiders make decisions
	at each point in time on where to move based on their expected fitness gain from moving to that location (we assume this expectation is exactly what their
	fitness would be if they moved to that location in the currenet time step). 

INPUTS:
- num_spiders: the number of spiders to begin the simulation with
- num_locations: this number is used to form a num_locations x num_locations grid where spiders live and move about.
- xi: the cost of travel
- kappa: the constant energy loss each period
- T: number of time steps to simulate
- arrival_rate: the poisson arrival rate of prey (one-calorie mosquitos)
- numMales: the number of males to include in the model that live on the webs of females and may either stay on a web or leave and move to a different web.
"""

### CONSTANTS
# Number of (female) spiders to begin with
num_spiders = 100
# Use a num_location x num_locations grid of locations for spiders to move amongth
num_locations = 5
# Cost of travel
xi = 2
# Constant energy loss each time step
kappa = 1
# Number of days to simulate
T = 100
# Poisson arrival rate of foot
arrival_rate = 5
# Number of males 
numMales=num_spiders

### FUNCTIONS
def number_at_location(row,col, location):
	""" Computes number of spiders at location (row, col). location is the slice of locations_panel at the appropriate time step """
	num = 0
	for j in range(num_spiders):
		if (location[j,0]==row and location[j,1]==col and location[j,0]>=0):
			num = num + 1
	return num

def size(F_panel, j, t):
	""" Computes size of spider j by taking weighted sum of all previous fitnesses from tau=1, ..., t (where t is the current period) """
	sum = 1
	for tau in range(t):
		sum = sum + (3/t)*((4*(t/2-tau)**2)/t**2)*F_panel[j,tau]
	return sum 

def dist(old_loc, new_loc):
	"""Comutes distance between two locations"""
	return ((old_loc[0]-new_loc[0])**2+(old_loc[1]-new_loc[1])**2)**.5

def fitness_func(newLoc, spider, F_panel, t, location, locations_of_others, sizes_crosssection):
	"""Computes fitness function for spider
		- newLoc is location to evaluate fitness at
		- spider is the spider unravel_index
		- F is the matrix of fitnesses
		- t is the current time period
		- location is the spider's current locations_panel
		- locations_of_others are the locations of all other spiders"""
	if (location[0]<0):
		return -20
	else:
		# newLoc[0] is the row (first element in tuple)
		newLoc_row = newLoc[0] 
		#newLoc[1] is the column (second element in tuple)
		newLoc_col = newLoc[1]
		return size(F_panel, spider, t)/v(newLoc_row,newLoc_col, t, spider, 
			locations_of_others, sizes_crosssection)*((u[newLoc_row][newLoc_col])) - xi*dist(location,newLoc) - kappa

def v(row,col, t, k, location, sizes_crosssection):
	"""Computes mass of spiders at a location (the sum of the current fitnesses of all spiders"""
	mass = 1
	for j in range(num_spiders):
		if (j!=k and location[j,0]==row and location[j,1]==col and F_panel[j,t]>=0):
			#print(sizes_crosssection[j])
			mass = mass + sizes_crosssection[j]
	return mass

def rargmax(vector):
    """ Argmax that chooses randomly among eligible maximum indices. """
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return random.choice(indices)

### INITIALIZE ALL ARRAYS/PANELS
# Initialize locations_panel: num_spiders x T matrix in which rows correspond to spiders and columns correspond to time periods.
locations_panel = [[(1,1) for t in range(T)] for j in range(num_spiders)]
# Randomly assign all spiders a position
for j in range(num_spiders):
	locations_panel[j][0] = (random.randint(0,num_locations-1), random.randint(0,num_locations-1))
locations_panel = np.array(locations_panel)

#male_locations = [(random.randint(0,num_locations-1), random.randint(0,num_locations-1)) for i in range(numMales)]

# Initialize F_panel: num_spiders x T matrix in which rows correspond to spiders and columns correspond to time periods. 
# Entries are the fitness of spider j at time t.
F_panel = [np.zeros(T) for i in range(num_spiders)]
F_panel = np.array(F_panel)


# Initialize sizes_panel: num_spiders x T matrix in which rows correspond to spiders and columns correspond to time periods. 
# Entries are the size of spider j at time t.
sizes_panel = [np.ones(T) for j in range(num_spiders)]
sizes_panel = np.array(sizes_panel)

# Initialize number_at_location_time_series where each entry corresponds to a time step and represents a spatial disribution of spiders 
# (in array n_rows x n_cols form)
number_at_location_time_series = [np.zeros((num_locations,num_locations)) for i in range(T)]
number_at_location_time_series = np.array(number_at_location_time_series)
# Count number at each location for t=0
for row in range(num_locations):
	for col in range(num_locations):
		number_at_location_time_series[0] = number_at_location(row,col,locations_panel[:,0])


#### MAIN LOOP
for t in range(T):
	# Generate food
	u = [[np.random.poisson(arrival_rate) for i in range(num_locations)] for k in range(num_locations)]
	#u = [[0 for i in range(num_locations)] for k in range(num_locations)]
	#u[1][1] = 100
	print(t)
	for j in range(num_spiders):
		# If alive
		if (locations_panel[j,t,0]>=0):
			# Spiders eat. Update fitness and size at time t
			F_panel[j,t] = fitness_func(locations_panel[j,t], j, F_panel, t, locations_panel[j,t],locations_panel[:,t], sizes_panel[:,t])
			sizes_panel[j,t] = size(F_panel, j, t)
			if (t+1<T):
				# After eating and updating size, if the the spider's size is less than 0, spider dies.
				if (sizes_panel[j,t] < 0.0):
					locations_panel[j,t+1, 0]=-99
					locations_panel[j,t+1, 1]=-99
				# If spider still alive, calculate fitness of potential moves and move to location providing highest expected fitness
				else:
					# Spiders calculate fitness from other locations
					F_calc = [[fitness_func((i, k), j, F_panel, t, locations_panel[j,t],locations_panel[:,t], sizes_panel[:,t]) for i in range(num_locations)] for k in range(num_locations)]
					F_calc = np.array(F_calc)
					# Save the coordinates of the location with highest expected fitness
					k,l = np.unravel_index(np.argmax(F_calc), F_calc.shape)
					# Unless at last time step, update locations_panel for where the spider moved
					locations_panel[j,t+1, 0] = k
					locations_panel[j,t+1, 1] = l
		# If already dead
		else:
			if (t+1<T):
				F_panel[j,t] = -20 
				sizes_panel[j,t] = 0
				locations_panel[j,t+1, 0]=-99
				locations_panel[j,t+1, 1]=-99

	# Count number of spiders at each location and save the count
	for row in range(num_locations):
		for col in range(num_locations):
			number_at_location_time_series[t][row,col] = number_at_location(row,col,locations_panel[:,t])
	

# Prints output
for t in range(T):
	print(t)
	print(number_at_location_time_series[t])
	print(locations_panel[1,t,0])
	print(sizes_panel[1,t])


with h5py.File('locationNum.h5', 'w') as hf:
    hf.create_dataset("number_at_location_time_series",  data=number_at_location_time_series)

with h5py.File('size.h5', 'w') as hf:
	hf.create_dataset("sizes_panel",  data=sizes_panel)

with h5py.File('locations.h5', 'w') as hf:
	hf.create_dataset("locations_panel",  data=locations_panel)
