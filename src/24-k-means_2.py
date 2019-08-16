import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

df = pd.read_excel('chosen123456.xlsx')
df.loc[df['most_often'].isna(),'most_often']=0

print(df.head())

features = df.columns[0:]

scaler = MinMaxScaler().fit(df[features])
x = pd.DataFrame(scaler.transform(df[features]))
x.columns = features


colors = ['red', 'green', 'blue', 'gold']


for i in [7]:
    print("Clusters:%d"%i)
    est = KMeans(n_clusters=i, init='random')
    est.fit(x)
    df['labels'] = est.labels_

    centers = pd.DataFrame(scaler.inverse_transform(est.cluster_centers_), columns=features)
    print('centers', centers)

print(centers.round())

