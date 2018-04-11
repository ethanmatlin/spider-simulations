import spider_simulation
import numpy as np
import numpy.matlib

# BASELINE MODEL
meta_parameters = {'num_spiders': 20,
						'num_locations': 20,
						'xi': 1, 
						'kappa': .5, 
						'T': 200, 
						'arrival_rate': 20, 
						'size_continuous_bool': True,
						'time_variant_arrival_rate': False, 
						'space_variant_arrival_rate': False,
						'space_variant_arrival_rate_mat': np.array([[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]),
						'space_variant_arrival_rate_lb': 10,
						'space_variant_arrival_rate_ub': 30,
						'omnicient': False, 
						'random_movement': True, 
						'random_movement_sd': 1,
						'smooth': False,
						'all_at_once': True,
						'start_clusters': False
					}

spider_simulation.simulate(meta_parameters)
spider_simulation.graph(meta_parameters)


meta_parameters = {'num_spiders': 20,
						'num_locations': 20,
						'xi': 1, 
						'kappa': .5, 
						'T': 200, 
						'arrival_rate': 20, 
						'size_continuous_bool': True,
						'time_variant_arrival_rate': False, 
						'space_variant_arrival_rate': False,
						'space_variant_arrival_rate_mat': np.array([[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]),
						'space_variant_arrival_rate_lb': 10,
						'space_variant_arrival_rate_ub': 30,
						'omnicient': False, 
						'random_movement': True, 
						'random_movement_sd': 1,
						'smooth': False,
						'all_at_once': True,
						'start_clusters': True
					}

spider_simulation.simulate(meta_parameters)
spider_simulation.graph(meta_parameters)


meta_parameters = {'num_spiders': 20,
						'num_locations': 20,
						'xi': 1, 
						'kappa': .5, 
						'T': 200, 
						'arrival_rate': 20, 
						'size_continuous_bool': True,
						'time_variant_arrival_rate': False, 
						'space_variant_arrival_rate': False,
						'space_variant_arrival_rate_mat': np.array([[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]),
						'space_variant_arrival_rate_lb': 10,
						'space_variant_arrival_rate_ub': 30,
						'omnicient': False, 
						'random_movement': True, 
						'random_movement_sd': 1,
						'smooth': False,
						'all_at_once': False,
						'start_clusters': False
					}

spider_simulation.simulate(meta_parameters)
spider_simulation.graph(meta_parameters)



# SPATIAL VARIATION IN ARRIVAL RATES
meta_parameters = {'num_spiders': 20,
						'num_locations': 20,
						'xi': 1, 
						'kappa': .5, 
						'T': 200, 
						'arrival_rate': 20, 
						'size_continuous_bool': True,
						'time_variant_arrival_rate': False, 
						'space_variant_arrival_rate': True,
						'space_variant_arrival_rate_mat': numpy.matlib.repmat(np.linspace(1,39,20),20,1),
						'space_variant_arrival_rate_lb': 10,
						'space_variant_arrival_rate_ub': 30,
						'omnicient': False, 
						'random_movement': True, 
						'random_movement_sd': 1,
						'smooth': False,
						'all_at_once': True,
						'start_clusters': False
					}
spider_simulation.simulate(meta_parameters)
spider_simulation.graph(meta_parameters)


# TEMPORAL VARIATION IN ARRIVAL RATES
meta_parameters = {'num_spiders': 20,
						'num_locations': 20,
						'xi': 1, 
						'kappa': .5, 
						'T': 200, 
						'arrival_rate': 20, 
						'size_continuous_bool': True,
						'time_variant_arrival_rate': True, 
						'space_variant_arrival_rate': False,
						'space_variant_arrival_rate_mat': np.array([[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]),
						'space_variant_arrival_rate_lb': 10,
						'space_variant_arrival_rate_ub': 30,
						'omnicient': False, 
						'random_movement': True, 
						'random_movement_sd': 1,
						'smooth': False,
						'all_at_once': True,
						'start_clusters': False
					}
spider_simulation.simulate(meta_parameters)
spider_simulation.graph(meta_parameters)

# Discrete Size (Percentiles)
meta_parameters = {'num_spiders': 20,
						'num_locations': 20,
						'xi': 1, 
						'kappa': .5, 
						'T': 200, 
						'arrival_rate': 20, 
						'size_continuous_bool': False,
						'time_variant_arrival_rate': False, 
						'space_variant_arrival_rate': False,
						'space_variant_arrival_rate_mat': np.array([[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]),
						'space_variant_arrival_rate_lb': 10,
						'space_variant_arrival_rate_ub': 30,
						'omnicient': False, 
						'random_movement': True, 
						'random_movement_sd': 1,
						'smooth': False,
						'all_at_once': True,
						'start_clusters': False
					}
spider_simulation.simulate(meta_parameters)
spider_simulation.graph(meta_parameters)

# Turn ON Omnicience
meta_parameters = {'num_spiders': 20,
						'num_locations': 20,
						'xi': 1, 
						'kappa': .5, 
						'T': 200, 
						'arrival_rate': 20, 
						'size_continuous_bool': True,
						'time_variant_arrival_rate': False, 
						'space_variant_arrival_rate': False,
						'space_variant_arrival_rate_mat': np.array([[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]),
						'space_variant_arrival_rate_lb': 10,
						'space_variant_arrival_rate_ub': 30,
						'omnicient': True, 
						'random_movement': True, 
						'random_movement_sd': 1,
						'smooth': False,
						'all_at_once': True,
						'start_clusters': False
					}
spider_simulation.simulate(meta_parameters)
spider_simulation.graph(meta_parameters)


# Turn ON Rationality (random on by default)
meta_parameters = {'num_spiders': 20,
						'num_locations': 20,
						'xi': 1, 
						'kappa': .5, 
						'T': 200, 
						'arrival_rate': 20, 
						'size_continuous_bool': True,
						'time_variant_arrival_rate': False, 
						'space_variant_arrival_rate': False,
						'space_variant_arrival_rate_mat': np.array([[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]),
						'space_variant_arrival_rate_lb': 10,
						'space_variant_arrival_rate_ub': 30,
						'omnicient': False, 
						'random_movement': False, 
						'random_movement_sd': 0,
						'smooth': False,
						'all_at_once': True,
						'start_clusters': False
					}
spider_simulation.simulate(meta_parameters)
spider_simulation.graph(meta_parameters)

# Turn on Smoothing
meta_parameters = {'num_spiders': 20,
						'num_locations': 20,
						'xi': 1, 
						'kappa': .5, 
						'T': 200, 
						'arrival_rate': 20, 
						'size_continuous_bool': True,
						'time_variant_arrival_rate': False, 
						'space_variant_arrival_rate': False,
						'space_variant_arrival_rate_mat': np.array([[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]),
						'space_variant_arrival_rate_lb': 10,
						'space_variant_arrival_rate_ub': 30,
						'omnicient': False, 
						'random_movement': True, 
						'random_movement_sd': 1,
						'smooth': True,
						'all_at_once': True,
						'start_clusters': False
					}
spider_simulation.simulate(meta_parameters)
spider_simulation.graph(meta_parameters)