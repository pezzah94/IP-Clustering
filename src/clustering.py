import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


df = pd.read_csv('dogs.csv');

features = df.columns[1:].tolist();


print(df[features])
scaler = MinMaxScaler().fit(df[features])
x = pd.DataFrame(scaler.transform(df[features])) #zasto se ovo radi?
#i kako radi fit(x), sa transformisanim vrednostima

x.columns = features;

colors = ['green', 'blue', 'red', 'orange']

fig = plt.figure(figsize=(5,5));
plt_ind=1;

for i in [3,4]:
    est = KMeans(n_clusters=i, init='random');
    est.fit(x);

    df['labels'] = est.labels_;
    print(df['labels']);
   # print(df.loc[df['labels']==0]);
    print(est.cluster_centers_)
    centers = pd.DataFrame(scaler.inverse_transform(est.cluster_centers_),columns=features); #dohvatamo tacne vrednosti centroida

    print('Original centroidi', centers); #original, nisu transformisani centroidi

    sp = fig.add_subplot(2, 1, plt_ind)

    for j in range(0,i):
        cluster = df.loc[df['labels']==j];
        plt.scatter(cluster['height'], cluster['weight'], color=colors[j], label="cluster %d"%j); #ucrtavanje tacaka na plot

    sp.scatter(centers['height'], centers['weight'], color='black', marker='x', label='centroidi');#crtanje centroida
    plt.title('Senka %0.3f' % silhouette_score(x, est.labels_));
    plt.legend()

    plt_ind += 1;

plt.tight_layout();
plt.show();