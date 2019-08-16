import pandas as pd

import numpy as np
from kmodes.kmodes import KModes



df = pd.read_excel('chosen.xlsx')

#izdvojim bitne kolone

print(df.columns)
f = ['Q16RPython', 'Q6', 'Q19_Part_1', 'Q19_Part_2', 'Q19_Part_3','Q13_Part_1', 'Q13_Part_2', 'Q13_Part_3']


df.loc[df['Q19_Part_1'].isna(),'Q19_Part_1']=''
df.loc[df['Q19_Part_2'].isna(),'Q19_Part_2']=''
df.loc[df['Q19_Part_3'].isna(),'Q19_Part_3']=''

df.loc[df['Q13_Part_1'].isna(),'Q13_Part_1']=''
df.loc[df['Q13_Part_2'].isna(),'Q13_Part_2']=''
df.loc[df['Q13_Part_3'].isna(),'Q13_Part_3']=''


df.loc[df['Q6'].isna(),'Q6']=''


data = df[f]
print(data)


"""
columns = ['Q4','Q6', 'Q7']

q16=['Q16_Part_%d'%i for i in range(1,16)]

print(q16)
q19=['Q19_Part_%d'%i for i in range(1,19)]
print(q19)
colnames = ['gender', 'education', 'current_role', 'emp_status']

print(df[q16]);

"""



km = KModes(n_clusters=7, init='Huang', n_init=5, verbose=1)

clusters = km.fit_predict(data)

# Print the cluster centroids
print(km.cluster_centroids_)



