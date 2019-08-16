import pandas as pd

import numpy as np
from kmodes.kmodes import KModes



df = pd.read_excel('chosen.xlsx')

#izdvojim bitne kolone

print(df.columns)
f = ['Q16RPython', 'Q7', 'Q6', 'Q4', 'Q17']
df.loc[df['Q17'].isna(),'Q17']=''
df.loc[df['Q7'].isna(),'Q7']=''
df.loc[df['Q6'].isna(),'Q6']=''
df.loc[df['Q4'].isna(),'Q4']=''

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



km = KModes(n_clusters=6, init='Huang', n_init=5, verbose=1)

clusters = km.fit_predict(data)

# Print the cluster centroids
print(km.cluster_centroids_)



