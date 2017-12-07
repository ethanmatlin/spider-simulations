import numpy as np 
import h5py 
import matplotlib.pyplot as plt
import pandas as pd



#####READING IN OUTPUTS
sizes = h5py.File("size.h5", 'r')
sizes = list(sizes["sizes_panel"])
sizes = np.array(sizes)

locationNum = h5py.File("locationNum.h5", 'r')
locationNum = list(locationNum["number_at_location_time_series"])
locationNum = np.array(locationNum)

locations = h5py.File("locations.h5", 'r')
locations = list(locations["locations_panel"])
locations = np.array(locations)


numLocations = len(locations[1])

def sdAtLoc(loc_y, loc_x, t):
	sdVec = np.zeros(len(locations[:,0]))
	for i in range(len(locations[:,0])):
		if (locations[i,t,0]==loc_y and locations[i,t,1]==loc_x):
			sdVec[i] = sizes[i,t]
	return np.std(sdVec)

def intracluster_sd(t):
	SDs = np.zeros((len(locations[:,0]), len(locations[:,1])))
	for y in range(len(locations[:,0])):
		for x in range(len(locations[:,1])):
			SDs[y,x] = sdAtLoc(y,x,t)
	return np.mean(SDs)

means = np.ones(len(sizes[0,:]))
intra_cluster_sds = np.ones(len(sizes[0,:]))
for t in range(len(sizes[1])):
	means[t] = np.mean(sizes[(sizes>0)[:,t],t])
	intra_cluster_sds[t] = intracluster_sd(t)


t=50
df = pd.DataFrame({'loc_y': locations[:,t,0], 'loc_x': locations[:,t,1], 'size': sizes[:,t]})
df = df[df['loc_x']>=0]

nreps=10000
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

plt.hist(nullSD)
plt.axvline(model, color='b', linestyle='dashed', linewidth=2)
plt.savefig('model_perm_test.png', bbox_inches='tight')


plt.plot(means)
plt.savefig('size_over_time.png', bbox_inches='tight')
plt.close()

plt.plot(intra_cluster_sds)
plt.savefig('intracluster_sd_over_time.png', bbox_inches='tight')
plt.close()

plt.imshow(locationNum[0], cmap='hot', interpolation='nearest')
plt.savefig('heatmap_t=0.png', bbox_inches='tight')
plt.close()

plt.imshow(locationNum[25], cmap='hot', interpolation='nearest')
plt.savefig('heatmap_t=25.png', bbox_inches='tight')
plt.close()

plt.imshow(locationNum[50], cmap='hot', interpolation='nearest')
plt.savefig('heatmap_t=50.png', bbox_inches='tight')
plt.close()

plt.imshow(locationNum[75], cmap='hot', interpolation='nearest')
plt.savefig('heatmap_t=75.png', bbox_inches='tight')
plt.close()

t = len(sizes[1])-1
plt.imshow(locationNum[t], cmap='hot', interpolation='nearest')
plt.savefig('heatmap_t='+str(t)+'.png', bbox_inches='tight')
plt.close()


plt.hist(sizes[:,0])
plt.savefig('hist_t=0.png', bbox_inches='tight')
plt.close()
plt.hist(sizes[:,25])
plt.savefig('hist_t=25.png', bbox_inches='tight')
plt.close()
plt.hist(sizes[:,50])
plt.savefig('hist_t=50.png', bbox_inches='tight')
plt.close()
plt.hist(sizes[:,75])
plt.savefig('hist_t=75.png', bbox_inches='tight')
plt.close()
plt.hist(sizes[:,99])
plt.savefig('hist_t=99.png', bbox_inches='tight')
plt.close()



# sdMat = np.ones((numLocations,numLocations))
# #for t in range(len(sizes[1])):
# for k in range(numLocations):
# 	for l in range(numLocations):
# 		sdMat[k,l] = sdAtLoc(k,l,49)

# print(sdMat)
# print(np.mean(sdMat))



# mat = np.ones((len(sizes), len(sizes[0])))

# for t in range(len(sizes[0])):
# 	for i in range(len(sizes)):
# 		print(t)
# 		print(i)
# 		mat[i,t] = sizes[i][t]

# for i in range(mat.shape):
# 	plt.plot(mat[i,])
# plt.show()

