import numpy as np 
import random
import plotly
import plotly.plotly as py
import time
import plotly.graph_objs as go
import h5py
import time 
import matplotlib.pyplot as plt
import pandas as pd


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
	"""Computes distance between two locations"""
	return ((old_loc[0]-new_loc[0])**2+(old_loc[1]-new_loc[1])**2)**.5

def fitness_func(newLoc, spider, F_panel, t, location, locations_of_others, sizes_crosssection, u):
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
			locations_of_others, sizes_crosssection, num_spiders, F_panel)*((u[newLoc_row][newLoc_col])) - xi*dist(location,newLoc) - kappa

def v(row,col, t, k, location, sizes_crosssection, num_spiders, F_panel):
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


def simulate(meta_parameters):
	### CONSTANTS
	# Number of (female) spiders to begin with
	num_spiders = meta_parameters['num_spiders']
	# Use a num_location x num_locations grid of locations for spiders to move amongth
	num_locations = meta_parameters['num_locations']
	# Cost of travel
	xi = meta_parameters['xi']
	# Constant energy loss each time step
	kappa = meta_parameters['kappa']
	# Number of days to simulate
	T = meta_parameters['T']
	# Poisson arrival rate of food
	arrival_rate = meta_parameters['arrival_rate']
	# If true, leaves the arrival rate at 5 for all time periods and location. If false, the arrival rate changes at each time step.
	constant_arrival_rate = meta_parameters['constant_arrival_rate']
	# If false, spiders can only see directly neighboring locations
	omnicient = meta_parameters['omnicient']
	# Random movement
	random_movement = meta_parameters['random_movement']
	random_movement_sd = meta_parameters['random_movement_sd']
	# Number of males 
	numMales=num_spiders

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
		if (not constant_arrival_rate):
			#arrival_rate = arrival_rate + np.random.normal
			arrival_rate = np.random.uniform(2.5,7.5)

		u = [[np.random.poisson(arrival_rate) for i in range(num_locations)] for k in range(num_locations)]
		#u = [[0 for i in range(num_locations)] for k in range(num_locations)]
		#u[1][1] = 100
		print(t)
		for j in range(num_spiders):
			# If alive
			if (locations_panel[j,t,0]>=0):
				# Spiders eat. Update fitness and size at time t
				F_panel[j,t] = fitness_func(locations_panel[j,t], j, F_panel, t, locations_panel[j,t],locations_panel[:,t], sizes_panel[:,t], u)
				sizes_panel[j,t] = size(F_panel, j, t)
				if (t+1<T):
					# After eating and updating size, if the the spider's size is less than 0, spider dies.
					if (sizes_panel[j,t] < 0.0):
						locations_panel[j,t+1, 0]=-99
						locations_panel[j,t+1, 1]=-99
					# If spider still alive, calculate fitness of potential moves and move to location providing highest expected fitness
					else:
						# Spiders calculate fitness from other locations
						# Consider all locations
						F_calc = [[fitness_func((i, k), j, F_panel, t, locations_panel[j,t],locations_panel[:,t], sizes_panel[:,t], u) for i in range(num_locations)] for k in range(num_locations)]
						F_calc = np.array(F_calc)
						# Set out of range locations to -99 
						if (omnicient==False):
							for row in range(num_locations):
								for col in range(num_locations):
									if (row > (locations_panel[j,t, 0]+1) or row < (locations_panel[j,t, 0] -1) or col > (locations_panel[j,t, 1]+1) or col < (locations_panel[j,t, 1]-1)):
										F_calc[row,col] = -99
						if (random_movement == True):
							for row in range(num_locations):
								for col in range(num_locations):
									F_calc[row,col] = F_calc[row,col] + np.random.normal(0, random_movement_sd)
						# Save the coordinates of the location with highest expected fitness
						k,l = np.unravel_index(np.argmax(F_calc), F_calc.shape)
						# Unless at last time step, update locations_panel for where the spider moved
						locations_panel[j,t+1, 0] = k
						locations_panel[j,t+1, 1] = l
					# if (j==1):
					# 	print(F_calc)
					# 	print(locations_panel[j,t+1])
					# 	print(sizes_panel[j,t])

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


	t = time.time()

	with h5py.File('locationNum.h5', 'w') as hf:
	    hf.create_dataset("number_at_location_time_series",  data=number_at_location_time_series)

	with h5py.File('size.h5', 'w') as hf:
		hf.create_dataset("sizes_panel",  data=sizes_panel)

	with h5py.File('locations.h5', 'w') as hf:
		hf.create_dataset("locations_panel",  data=locations_panel)

	print("Time to save data: " + str(time.time() - t))


def graph(meta_parameters):
	sizes_panel = h5py.File("size.h5", 'r')
	sizes_panel = list(sizes_panel["sizes_panel"])
	sizes_panel = np.array(sizes_panel)

	number_at_location_time_series = h5py.File("locationNum.h5", 'r')
	number_at_location_time_series = list(number_at_location_time_series["number_at_location_time_series"])
	number_at_location_time_series = np.array(number_at_location_time_series)

	locations_panel = h5py.File("locations.h5", 'r')
	locations_panel = list(locations_panel["locations_panel"])
	locations_panel = np.array(locations_panel)

	numLocations = len(locations_panel[1])

	def sdAtLoc(loc_y, loc_x, t):
		sizes_at_location = np.zeros(len(locations_panel[:,0]))
		for i in range(len(locations_panel[:,0])):
			if (locations_panel[i,t,0]==loc_y and locations_panel[i,t,1]==loc_x and sizes_panel[i,t]>0):
				sizes_at_location[i] = sizes_panel[i,t]
		return np.std(sizes_at_location)

	def intracluster_sd(t):
		SDs = np.zeros((len(locations_panel[:,0]), len(locations_panel[:,1])))
		for y in range(len(locations_panel[:,0])):
			for x in range(len(locations_panel[:,1])):
				SDs[y,x] = sdAtLoc(y,x,t)
		return np.mean(SDs)

	t = time.time()
	means = np.ones(len(sizes_panel[0,:]))
	intra_cluster_sds = np.ones(len(sizes_panel[0,:]))
	num_dead = np.ones(len(sizes_panel[0,:]))
	for t in range(len(sizes_panel[1])):
		means[t] = np.mean(sizes_panel[(sizes_panel>0)[:,t],t])
		num_dead[t] = len(sizes_panel[(sizes_panel<=0)[:,t],t])
		intra_cluster_sds[t] = intracluster_sd(t)

	print("Time to calculate intra-cluster SDs: " + str(time.time()-t))

	t=50
	df = pd.DataFrame({'loc_y': locations_panel[:,t,0], 'loc_x': locations_panel[:,t,1], 'size': sizes_panel[:,t]})
	df = df[df['loc_x']>=0]

	t = time.time()
	nreps=1000
	nullSD = np.zeros(nreps)
	model = 0
	for i in range(nreps):
		df['size_perm'] = np.random.permutation(df['size'])
		grouped = df.groupby(['loc_x', 'loc_y'])
		model = np.mean(grouped.aggregate(np.std))[0]
		nullSD[i] = np.mean(grouped.aggregate(np.std))[1]

	pval = np.sum(nullSD<model)/nreps
	print("Model avg. intracluster SD: " + str(model))
	print("Null avg. intracluster SD: " + str(np.mean(nullSD)))
	print("p-value: " + str(pval))

	print("Time for Permutation test:" + str(time.time()-t))

	plt.hist(nullSD)
	plt.axvline(model, color='b', linestyle='dashed', linewidth=2)
	plt.savefig('model_perm_test.png', bbox_inches='tight')
	plt.close()

	plt.plot(means)
	plt.savefig('size_over_time.png', bbox_inches='tight')
	plt.close()

	plt.plot(intra_cluster_sds)
	plt.savefig('intracluster_sd_over_time.png', bbox_inches='tight')
	plt.close()

	plt.imshow(number_at_location_time_series[0], cmap='hot', interpolation='nearest')
	plt.savefig('heatmap_t=0.png', bbox_inches='tight')
	plt.close()

	plt.imshow(number_at_location_time_series[25], cmap='hot', interpolation='nearest')
	plt.savefig('heatmap_t=25.png', bbox_inches='tight')
	plt.close()

	plt.imshow(number_at_location_time_series[50], cmap='hot', interpolation='nearest')
	plt.savefig('heatmap_t=50.png', bbox_inches='tight')
	plt.close()

	plt.imshow(number_at_location_time_series[75], cmap='hot', interpolation='nearest')
	plt.savefig('heatmap_t=75.png', bbox_inches='tight')
	plt.close()

	t = len(sizes_panel[1])-1
	plt.imshow(number_at_location_time_series[t], cmap='hot', interpolation='nearest')
	plt.savefig('heatmap_t='+str(t)+'.png', bbox_inches='tight')
	plt.close()


	plt.hist(sizes_panel[:,0])
	plt.savefig('hist_t=0.png', bbox_inches='tight')
	plt.close()
	plt.hist(sizes_panel[:,25])
	plt.savefig('hist_t=25.png', bbox_inches='tight')
	plt.close()
	plt.hist(sizes_panel[:,50])
	plt.savefig('hist_t=50.png', bbox_inches='tight')
	plt.close()
	plt.hist(sizes_panel[:,75])
	plt.savefig('hist_t=75.png', bbox_inches='tight')
	plt.close()
	plt.hist(sizes_panel[:,99])
	plt.savefig('hist_t=99.png', bbox_inches='tight')
	plt.close()


meta_parameters = {'num_spiders': 100,
						'num_locations': 5,
						'xi': xi, 
						'kappa': 2, 
						'T': 100, 
						'arrival_rate': 10, 
						'constant_arrival_rate': False, 
						'omnicient': True, 
						'random_movement': False, 
						'random_movement_sd': 1

simulate(meta_parameters)
graph(meta_parameters)

