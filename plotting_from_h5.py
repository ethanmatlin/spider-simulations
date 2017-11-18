import numpy as np 
import h5py 
import matplotlib.pyplot as plt


def numAtLoc(loc_y,loc_x):
	counter = 0
	for i in range(len(locations)):
		if (locations[i][0]==loc_y and locations[i][1]==loc_x):
			counter = counter + 1
	return counter

def sdAtLoc(loc_y, loc_x, t):
	sdVec = []
	for i in range(len(locations[0])):
		if (locations[t][i][0]==loc_y and locations[t][i][1]==loc_x):
			#print(sizes[i][t])
			sdVec.append(sizes[i][t])
			print(sdVec)
	#print(sdVec)
	return np.std(sdVec)

#####READING IN OUTPUTS
sizes = h5py.File("size.h5", 'r')
sizes = list(sizes["sizeMat"])

locationNum = h5py.File("locationNum.h5", 'r')
locationNum = list(locationNum["allMats"])

locations = h5py.File("locations.h5", 'r')
locations = list(locations["locations"])

numLocations = len(locations[1])


means = np.ones(50)

for t in range(len(sizes[1])):
	sum = 0 
	for i in range(len(sizes)):
		sum = sum + sizes[i][t]
	means[t] = sum/len(sizes)


# sdMat = np.ones((numLocations,numLocations))
# #for t in range(len(sizes[1])):
# for k in range(numLocations):
# 	for l in range(numLocations):
# 		sdMat[k,l] = sdAtLoc(k,l,49)

# print(sdMat)
# print(np.mean(sdMat))

plt.plot(means)
plt.show()

mat = np.ones((len(sizes), len(sizes[0])))

for t in range(len(sizes[0])):
	for i in range(len(sizes)):
		print(t)
		print(i)
		mat[i,t] = sizes[i][t]

for i in range(mat.shape):
	plt.plot(mat[i,])
plt.show()

