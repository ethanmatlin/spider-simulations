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
import mpl_toolkits.axes_grid1 as axes_grid1
import math 



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
def number_at_location(row,col, location, num_spiders):
	""" Computes number of spiders at location (row, col). location is the slice of locations_panel at the appropriate time step """
	num = 0
	for j in range(num_spiders):
		if (location[j,0]==row and location[j,1]==col and location[j,0]>=0):
			num = num + 1
	return num

def size_continuous(F_panel, j, t):
	""" Computes size of spider j by taking weighted sum of all previous fitnesses from tau=1, ..., t (where t is the current period) """
	#print(F_panel[j,(t-4):t])
	sum = 0
	for tau in range(1,t):
		#pringrt(F_panel[j,tau])
		## CHANGED TO 1/t INSTEAD TO FIX THE HUGE SPIKE PROBLEM	
		sum = sum + (1/t)*((4*(t/2-tau)**2)/t**2)*F_panel[j,tau]
	return sum 

def rescale_size(sizes_crosssection, num_spiders):
	rescaled_sizes = np.zeros(num_spiders)
	#     Large     Medium       Mini      Small 
	#	0.05155321 0.34765367 0.10905486 0.49173827 
	print(sizes_crosssection)
	print("95" + str(np.percentile(sizes_crosssection,95)))
	print("65" + str(np.percentile(sizes_crosssection,65)))
	print("11" + str(np.percentile(sizes_crosssection,11)))
	print("mean" + str(np.mean(sizes_crosssection)))
	for j in range(num_spiders):
		if (sizes_crosssection[j]>np.percentile(sizes_crosssection,95)):
			rescaled_sizes[j] = 4
		elif (sizes_crosssection[j]>np.percentile(sizes_crosssection,60)):
			rescaled_sizes[j] = 3
		elif (sizes_crosssection[j]>np.percentile(sizes_crosssection,11)):
			rescaled_sizes[j] = 2
		else: 
			rescaled_sizes[j] = 1
	#print(np.mean(rescaled_sizes))
	return rescaled_sizes 

 	#if sum(F_panel[j,(t-4):t] > )
	#if 

def dist(old_loc, new_loc):
	"""Computes distance between two locations"""
	return ((old_loc[0]-new_loc[0])**2+(old_loc[1]-new_loc[1])**2)**.5

def fitness_func(newLoc, spider, t, locations, sizes_crosssection, u, num_spiders, xi, kappa):
	"""Computes fitness function for spider
		- newLoc is location to evaluate fitness at
		- spider is the spider unravel_index
		- F is the matrix of fitnesses
		- t is the current time period
		- location is the spider's current locations_panel
		- locations_of_others are the locations of all other spiders"""
	if (locations[spider,t][0]<0):
		return -20
	else:
		# newLoc[0] is the row (first element in tuple)
		newLoc_row = newLoc[0] 
		#newLoc[1] is the column (second element in tuple)
		newLoc_col = newLoc[1]
		return 2*sizes_crosssection[spider]/v(newLoc_row,newLoc_col, t, spider, 
			locations[:,t], sizes_crosssection, num_spiders)*((u[newLoc_row][newLoc_col])) - xi*dist(locations[spider,t],newLoc) - kappa
		#return size_continuous(F_panel, spider, t)/v(newLoc_row,newLoc_col, t, spider, 
			#locations_of_others, sizes_crosssection, num_spiders, F_panel)*((u[newLoc_row][newLoc_col])) - xi*dist(location,newLoc) - kappa

def v(row,col, t, k, location, sizes_crosssection, num_spiders):
	"""Computes mass of spiders at a location (the sum of the current fitnesses of all spiders"""
	mass = 1
	for j in range(num_spiders):
		if (j!=k and location[j,0]==row and location[j,1]==col):
			#and F_panel[j,t]>=0
			#print(sizes_crosssection[j])
			mass = mass + 1
	# for j in range(num_spiders):
	# 	if (j!=k and location[j,0]==row and location[j,1]==col and F_panel[j,t]>=0):
	# 		#print(sizes_crosssection[j])
	# 		mass = mass + sizes_crosssection[j]
	# #print(mass)
	return mass

def rargmax(vector):
    """ Argmax that chooses randomly among eligible maximum indices. """
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return random.choice(indices)


def simulate(meta_parameters):
	### CONSTANTS
	num_spiders = meta_parameters['num_spiders']
	num_locations = meta_parameters['num_locations']
	xi = meta_parameters['xi']
	kappa = meta_parameters['kappa']
	T = meta_parameters['T']
	arrival_rate = meta_parameters['arrival_rate']
	time_variant_arrival_rate = meta_parameters['time_variant_arrival_rate']
	space_variant_arrival_rate = meta_parameters['space_variant_arrival_rate']
	space_variant_arrival_rate_mat = meta_parameters['space_variant_arrival_rate_mat']
	space_variant_arrival_rate_lb = meta_parameters['space_variant_arrival_rate_lb']
	space_variant_arrival_rate_ub = meta_parameters['space_variant_arrival_rate_ub']
	size_continuous_bool = meta_parameters['size_continuous_bool']
	omnicient = meta_parameters['omnicient']
	random_movement = meta_parameters['random_movement']
	random_movement_sd = meta_parameters['random_movement_sd']
	smooth = meta_parameters['smooth']
	all_at_once = meta_parameters['all_at_once']
	start_clusters = meta_parameters['start_clusters']
	numMales=num_spiders

	### INITIALIZE ALL ARRAYS/PANELS
	# Initialize locations_panel: num_spiders x T matrix in which rows correspond to spiders and columns correspond to time periods.
	locations_panel = [[(1,1) for t in range(T)] for j in range(num_spiders)]
	# Randomly assign all spiders a position
	if (start_clusters):
		for j in range(int(num_spiders/3)):
			locations_panel[j][0] = (random.randint(0,num_locations-1), random.randint(0,num_locations-1))
		locations_panel = np.array(locations_panel)
		for j in range(int(num_spiders/3),num_spiders):
			locations_panel[j][0] = random.choice(locations_panel[0:int(num_spiders/3),0])
	else:
		for j in range(int(num_spiders)):
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
			number_at_location_time_series[0] = number_at_location(row,col,locations_panel[:,0], num_spiders)


	F_calc_lag = np.array([[[0 for i in range(num_locations)] for k in range(num_locations)] for t in range(5)])
	#### MAIN LOOP
	for t in range(T):
		# Generate food
		if (time_variant_arrival_rate):
			#arrival_rate = arrival_rate + np.random.normal
			arrival_rate = np.random.uniform(12,28)
		if (space_variant_arrival_rate==False):
			u = [[np.random.poisson(arrival_rate) for i in range(num_locations)] for k in range(num_locations)]
		else: 
			u = [[np.random.poisson(space_variant_arrival_rate_mat[i,k]) for i in range(num_locations)] for k in range(num_locations)]
		for j in range(num_spiders):
			# If alive
			if (locations_panel[j,t,0]>=0):
				# Spiders eat. Update fitness and size at time t
				F_panel[j,t] = fitness_func(locations_panel[j,t], j, t, locations_panel, sizes_panel[:,t], u, num_spiders, xi, kappa)
				sizes_panel[j,t] = size_continuous(F_panel, j, t)
			# If already dead
			else:
				if (t+1<T):
					F_panel[j,t] = -20 
					sizes_panel[j,t] = 0
					locations_panel[j,t+1, 0]=-99
					locations_panel[j,t+1, 1]=-99
		if (t+1<T):
			locations_panel[:,t+1] = locations_panel[:,t] 
		for j in range(num_spiders):
			# Rescale sizes after all spiders have eaten
			if (size_continuous_bool==False):
					sizes_panel[:,t] = rescale_size(sizes_panel[:,t], num_spiders)
			if (t+1<T):
				# After eating and updating size, if the the spider's size is less than 0, spider dies.
				if (sizes_panel[j,t] < 0.0):
					locations_panel[j,t+1, 0]=-99
					locations_panel[j,t+1, 1]=-99
				# If spider still alive, calculate fitness of potential moves and move to location providing highest expected fitness
				else:
					F_calc_lag[4] = F_calc_lag[3]
					F_calc_lag[3] = F_calc_lag[2]
					F_calc_lag[2] = F_calc_lag[1]
					F_calc_lag[1] = F_calc_lag[0]
					# Spiders calculate fitness from other locations
					# Consider all locations
					if (all_at_once==True):
						F_calc = [[fitness_func((i, k), j, t, locations_panel, sizes_panel[:,t], u, num_spiders, xi, kappa) for i in range(num_locations)] for k in range(num_locations)]
						F_calc = np.array(F_calc)
						F_calc_lag[0] = F_calc
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
						if (smooth==False):
							k,l = np.unravel_index(np.argmax(F_calc), F_calc.shape)
						else:
							F_calc_avg = np.array([[np.mean(F_calc_lag[:,i,j]) for i in range(num_locations)] for j in range(num_locations)])
							k,l = np.unravel_index(np.argmax(F_calc_avg), F_calc_avg.shape)
					else:
						F_calc = [[fitness_func((i, k), j, t+1, locations_panel, sizes_panel[:,t+1], u, num_spiders, xi, kappa) for i in range(num_locations)] for k in range(num_locations)]
						F_calc = np.array(F_calc)
						F_calc_lag[0] = F_calc
						# Set out of range locations to -99 
						if (omnicient==False):
							for row in range(num_locations):
								for col in range(num_locations):
									if (row > (locations_panel[j,t+1, 0]+1) or row < (locations_panel[j,t+1, 0] -1) or col > (locations_panel[j,t+1, 1]+1) or col < (locations_panel[j,t+1, 1]-1)):
										F_calc[row,col] = -99
						if (random_movement == True):
							for row in range(num_locations):
								for col in range(num_locations):
									F_calc[row,col] = F_calc[row,col] + np.random.normal(0, random_movement_sd)
						# Save the coordinates of the location with highest expected fitness
						if (smooth==False):
							k,l = np.unravel_index(np.argmax(F_calc), F_calc.shape)
						else:
							F_calc_avg = np.array([[np.mean(F_calc_lag[:,i,j]) for i in range(num_locations)] for j in range(num_locations)])
							k,l = np.unravel_index(np.argmax(F_calc_avg), F_calc_avg.shape)

					# Unless at last time step, update locations_panel for where the spider moved
					locations_panel[j,t+1, 0] = k
					locations_panel[j,t+1, 1] = l

		# Count number of spiders at each location and save the count
		for row in range(num_locations):
			for col in range(num_locations):
				number_at_location_time_series[t][row,col] = number_at_location(row,col,locations_panel[:,t], num_spiders)
		

	# Prints output
	# for t in range(T):
	# 	print(t)
	# 	print(number_at_location_time_series[t])
	# 	print(locations_panel[1,t,0])
	# 	print(sizes_panel[1,t])


	t = time.time()

	with h5py.File('locationNum.h5', 'w') as hf:
	    hf.create_dataset("number_at_location_time_series",  data=number_at_location_time_series)

	with h5py.File('size.h5', 'w') as hf:
		hf.create_dataset("sizes_panel",  data=sizes_panel)

	with h5py.File('fitness.h5', 'w') as hf:
		hf.create_dataset("F_panel",  data=F_panel)

	with h5py.File('locations.h5', 'w') as hf:
		hf.create_dataset("locations_panel",  data=locations_panel)

	#print("Time to save data: " + str(time.time() - t))

	df1 = pd.DataFrame(sizes_panel)
	#print(np.array(df1.agg(['mean'])))

	df = pd.DataFrame(F_panel)
	#print(np.array(df.agg(['mean'])))

	#plt.plot(np.array(df.agg(['mean']))[0,:])
	#plt.show()


# df %>% groupby(col1) %>% summarize(col2_agg=max(col2), col3_agg=min(col3))
# is

# df.groupby('col1').agg({'col2': 'max', 'col3': 'min'})
# 	print(F_panel[:,t])

def numAt(loc_y, loc_x, t, locations_panel, sizes_panel):
	num_at = 0
	for i in range(len(locations_panel[:,0])):
		if (locations_panel[i,t,0]==loc_y and locations_panel[i,t,1]==loc_x and sizes_panel[i,t]>0):
			num_at = num_at + 1
	return num_at

def graph(meta_parameters):
	sizes_panel = h5py.File("size.h5", 'r')
	sizes_panel = list(sizes_panel["sizes_panel"])
	sizes_panel = np.array(sizes_panel)

	F_panel = h5py.File("fitness.h5", 'r')
	F_panel = list(F_panel["F_panel"])
	F_panel = np.array(F_panel)

	number_at_location_time_series = h5py.File("locationNum.h5", 'r')
	number_at_location_time_series = list(number_at_location_time_series["number_at_location_time_series"])
	number_at_location_time_series = np.array(number_at_location_time_series)

	locations_panel = h5py.File("locations.h5", 'r')
	locations_panel = list(locations_panel["locations_panel"])
	locations_panel = np.array(locations_panel)

	numLocations = locations_panel.shape[0]

	def sdAtLoc(loc_y, loc_x, t):
		sizes_at_location = np.zeros(len(locations_panel[:,0]))
		for i in range(len(locations_panel[:,0])):
			if (locations_panel[i,t,0]==loc_y and locations_panel[i,t,1]==loc_x and sizes_panel[i,t]>0):
				sizes_at_location[i] = sizes_panel[i,t]
		return np.std(sizes_at_location)

	
	def intracluster_sd(t):
		SDs = np.zeros((len(locations_panel[:,0]), len(locations_panel[:,1])))
		nums = np.zeros((len(locations_panel[:,0]), len(locations_panel[:,1])))
		for y in range(len(locations_panel[:,0])):
			for x in range(len(locations_panel[:,1])):
				SDs[y,x] = sdAtLoc(y,x,t)
				nums[y,x] = numAt(y,x,t, locations_panel, sizes_panel)
		return [np.mean(SDs), np.mean(nums[np.nonzero(nums)])]

	t = time.time()
	means = np.ones(len(sizes_panel[0,:]))
	mean_fitness = np.ones(len(F_panel[0,:]))
	intra_cluster_sds = np.ones(len(sizes_panel[0,:]))
	num_dead = np.ones(len(sizes_panel[0,:]))
	avg_num_at = np.ones(len(sizes_panel[0,:]))
	num_clusters = np.zeros(len(sizes_panel[0,:]))
	for t in range(len(sizes_panel[1])):
		means[t] = np.mean(sizes_panel[(sizes_panel>0)[:,t],t])
		mean_fitness[t] = np.mean(F_panel[(F_panel>0)[:,t],t])
		num_dead[t] = len(sizes_panel[(sizes_panel<=0)[:,t],t])
		intra_cluster_sds[t] = intracluster_sd(t)[0]
		avg_num_at[t] = np.mean(number_at_location_time_series[t][np.nonzero(number_at_location_time_series[t])])
		num_clusters[t] = np.count_nonzero(number_at_location_time_series[t])
		#avg_num_at[t] = intracluster_sd(t)[1]


	#print("Time to calculate intra-cluster SDs: " + str(time.time()-t))

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
	print("Time-Variant Arrival Rate: " + str(meta_parameters['time_variant_arrival_rate']))
	print("Omniscient: " + str(meta_parameters['omnicient']))
	print("Random Movement: " + str(str(meta_parameters['random_movement'])))
	print("Model avg. intracluster SD: " + str(model))
	print("Null avg. intracluster SD: " + str(np.mean(nullSD)))
	print("p-value: " + str(pval))
	print("Number dead: " + str(num_dead))
	print("Average number per location" + str(avg_num_at))
	print("")

	#print("Time for Permutation test:" + str(time.time()-t))

	ratio = meta_parameters['num_spiders']/(meta_parameters['num_locations'])**2
	if (meta_parameters['random_movement']==True):
		random_str = str(meta_parameters['random_movement_sd'])
	else: 
		random_str = str(0)

	def boolString(input):
		if (input==True):
			return 'T'
		else:
			return 'F'

	name_params = ("R=" + str(int(ratio)) + "sizecont=" + boolString(meta_parameters['size_continuous_bool']) 
	+ "timevar=" + boolString(meta_parameters['time_variant_arrival_rate'])
	+ "spacevar=" + boolString(meta_parameters['space_variant_arrival_rate']) 
	+ "omni=" + boolString(meta_parameters['omnicient']) + "rand=" + str(random_str) 
	+ "smoo=" + boolString(meta_parameters['smooth']) + "once=" + boolString(meta_parameters['all_at_once']) + "sc=" + boolString(meta_parameters['start_clusters']))

	plt.hist(nullSD)
	plt.axvline(model, color='b', linestyle='dashed', linewidth=2)
	plt.savefig('model_perm_test' + name_params + '.png', bbox_inches='tight')
	plt.close()

	plt.plot(mean_fitness)
	plt.xlabel("Time")
	plt.xlabel("Fitness")
	plt.savefig('fitness_over_time' + name_params + '.png', bbox_inches='tight')
	plt.close()

	plt.plot(means)
	plt.xlabel("Time")
	plt.ylabel("Size")
	plt.savefig('size_over_time' + name_params + '.png', bbox_inches='tight')
	plt.close()

	plt.plot(avg_num_at)
	plt.xlabel("Time")
	plt.ylabel("Number per Cluster")
	plt.savefig('num_at_over_time' + name_params + '.png', bbox_inches='tight')
	plt.close()

	plt.plot(num_clusters)
	plt.xlabel("Time")
	plt.ylabel("Number of Clusters")
	plt.savefig('num_clusters' + name_params + '.png', bbox_inches='tight')
	plt.close()

	plt.plot(intra_cluster_sds)
	plt.xlabel("Time")
	plt.ylabel("Intra-cluster Size SD")
	plt.savefig('intracluster_sd_over_time' + name_params + '.png', bbox_inches='tight')
	plt.close()


	im = plt.imshow(number_at_location_time_series[len(sizes_panel[1])-1], interpolation='nearest', cmap = 'hot', extent=[0,numLocations,0,numLocations])
	clim=im.properties()['clim']
	plt.close()


	# for i in np.linspace(0, len(sizes_panel[1])-1, len(sizes_panel[1]-1)/24):
	# 	fig = plt.figure()
	# 	grid = axes_grid1.AxesGrid(fig, 111, nrows_ncols=(1, 1), axes_pad = 0.5, cbar_location = "right", 
	# 		cbar_mode="each", cbar_size="15%", cbar_pad="5%")
	# 	im = grid[0].imshow(number_at_location_time_series[int(i)], interpolation='nearest', cmap = 'hot', clim = clim, extent=[0,numLocations,0,numLocations])
	# 	grid.cbar_axes[0].colorbar(im)
	# 	plt.savefig('heatmap_t=' + str(int(i)) + name_params + '.png', bbox_inches='tight')
	# 	plt.close(fig)
	# 	plt.hist(sizes_panel[:,int(i)], range=(0,np.amax(sizes_panel)))
	# 	plt.savefig('hist_t=' + str(int(i)) + name_params + '.png', bbox_inches='tight')
	# 	plt.close()

	for i in range(0, len(sizes_panel[1])-1):
		fig = plt.figure()
		grid = axes_grid1.AxesGrid(fig, 111, nrows_ncols=(1, 1), axes_pad = 0.5, cbar_location = "right", 
			cbar_mode="each", cbar_size="15%", cbar_pad="5%")
		im = grid[0].imshow(number_at_location_time_series[int(i)], interpolation='nearest', cmap = 'hot', clim = clim, extent=[0,numLocations,0,numLocations])
		grid.cbar_axes[0].colorbar(im)
		fig.suptitle('t = ' + str(int(i)))
		plt.savefig('heatmap_' + name_params + 't=' + str(int(i)).zfill(3) + '.png', bbox_inches='tight')
		plt.close(fig)


# num_spiders: Number of (female) spiders to begin with
# num_locations: Use a num_location x num_locations grid of locations for spiders to move amongth
# xi: Cost of travel
# kappa: Constant energy loss each time step
# T: Number of days to simulate
# arrival_rate: Poisson arrival rate of food
# time_variant_arrival_rate: If true, the arrival rate changes at each time step.
# space_variant_arrival_rate
# space_variant_arrival_rate_lb
# space_variant_arrival_rate_ub
# omnicient: If false, spiders can only see directly neighboring locations
# random_movement: Random movement
# random_movement_sd


