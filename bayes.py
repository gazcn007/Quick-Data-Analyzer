import pandas as pd
from pandas import DataFrame
import numpy as np

# load the datasets
names = ['dealerId', 'gender']
df = DataFrame(pd.read_csv("./GenderPythonTest.csv", sep=',',  names=names))

# Probablistic network build-up
bayesNetwork = [[],[],[]]
counter = -1
for idx in df['dealerId'].index:
	featureName = df.get_value(idx,'dealerId')
	if featureName not in bayesNetwork[0]:
		if counter !=-1 and bayesNetwork[1][counter]+bayesNetwork[2][counter] == 0 :
			#extreme case, the data is fault, hence overwrite the previous feature
			bayesNetwork[0][counter]=featureName
		else:
			bayesNetwork[0].append(featureName)
			bayesNetwork[1].append(0);
			bayesNetwork[2].append(0);
			counter+=1
	gender = df.get_value(idx,'gender')
	if gender == 'male':
		bayesNetwork[1][counter]+=1
	elif gender == 'female':
		bayesNetwork[2][counter]+=1

# For visualization
viz_data = {'Feature':bayesNetwork[0],
	'Male Count':bayesNetwork[1],
	'Female_Count':bayesNetwork[2]
}
df2 = DataFrame(viz_data,columns=['Feature','Male_Count','Female_Count'])
# df2.plot(kind='bar',stacked=True).get_figure().savefig('./Visualizer.png')

# Save bayesNetwork to csv
bayes_data = {
	'Feature': bayesNetwork[0],
	'Probablities': [float(x)/float(x+y) for x,y in zip(bayesNetwork[1],bayesNetwork[2])]}
df3 = DataFrame(bayes_data,columns=['Feature','Probablities'])
df3.to_csv('./Bayes_Network.csv',sep=',')
# df.to_csv