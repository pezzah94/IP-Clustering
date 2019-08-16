import pandas as pd
from kmodes.kmodes import KModes

df = pd.read_excel('RPyq19.xlsx')
print(df.columns)

print(df.head())
f = ['Q16RPython', 'Q19_Part_1', 'Q19_Part_2', 'Q19_Part_3', 'Q19_Part_9', 'Q19_Part_10',
       'Q19_Part_13'];

#replace Nan values with empty space
df.loc[df['Q19_Part_1'].isna(),'Q19_Part_1']=''
df.loc[df['Q19_Part_2'].isna(),'Q19_Part_2']=''
df.loc[df['Q19_Part_3'].isna(),'Q19_Part_3']=''
df.loc[df['Q19_Part_9'].isna(),'Q19_Part_9']=''
df.loc[df['Q19_Part_10'].isna(),'Q19_Part_10']=''
df.loc[df['Q19_Part_13'].isna(),'Q19_Part_13']=''


#sort columns
data = df[f]


km = KModes(n_clusters=6, init='Huang', n_init=15)

#clustering data
clusters = km.fit_predict(data)

# Pretty printing cluster centroids
table = pd.DataFrame(km.cluster_centroids_)


#pretty print of full table
table.replace('', 'empty', inplace=True)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
     print(table)