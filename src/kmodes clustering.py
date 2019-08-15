import pandas as pd
from kmodes.kmodes import KModes

df = pd.read_excel('RPyIde.xlsx')
print(df.columns)

print(df.head())

#replace Nan values with empty space
df.loc[df['Q13_Part_1'].isna(),'Q13_Part_1']=''
df.loc[df['Q13_Part_2'].isna(),'Q13_Part_2']=''
df.loc[df['Q13_Part_3'].isna(),'Q13_Part_3']=''
df.loc[df['Q21_Part_1'].isna(),'Q21_Part_1']=''
df.loc[df['Q21_Part_2'].isna(),'Q21_Part_2']=''
df.loc[df['Q21_Part_4'].isna(),'Q21_Part_4']=''
df.loc[df['Q21_Part_6'].isna(),'Q21_Part_6']=''
df.loc[df['Q21_Part_8'].isna(),'Q21_Part_8']=''

#sort columns
f = ['Q16RPython',
     'Q13_Part_1', 'Q13_Part_2', 'Q13_Part_3',
     'Q21_Part_1', 'Q21_Part_2', 'Q21_Part_4', 'Q21_Part_6', 'Q21_Part_8']

data = df[f]


km = KModes(n_clusters=6, init='Huang', n_init=5, verbose=1)

#clustering data
clusters = km.fit_predict(data)

# Pretty printing cluster centroids
table = pd.DataFrame(km.cluster_centroids_)


#pretty print of full table
table.replace('', 'empty', inplace=True)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
     print(table)