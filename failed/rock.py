from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.rock import rock
from pyclustering.samples.definitions import FCPS_SAMPLES
from pyclustering.utils import read_sample
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
# Read sample for clustering from file.
df = pd.read_excel('chosen12345610percent.xlsx')

features = df.columns
print(features)
f = ['RPy', 'edu', 'role', 'industry', 'most_often']

#print(df.values)

df.loc[df['RPy'].isna(),'RPy']=0
df.loc[df['edu'].isna(),'edu']=0
df.loc[df['role'].isna(),'role']=0
df.loc[df['industry'].isna(),'industry']=0
df.loc[df['most_often'].isna(),'most_often']=0

data = df[features]

scaler = MinMaxScaler().fit(data);
x = pd.DataFrame(scaler.transform(data));

# Create instance of ROCK algorithm for cluster analysis. Seven clusters should be allocated.
rock_instance = rock(x.values, 1.0, 7)
# Run cluster analysis.
rock_instance.process()
# Obtain results of clustering.
clusters = rock_instance.get_clusters()

print(clusters)
# Visualize clustering results.
#visualizer = cluster_visualizer()
#visualizer.append_clusters(clusters, x.values)
#visualizer.show()