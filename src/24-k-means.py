import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

df = pd.read_excel('processed data/chosen123456.xlsx')



features = df.columns[1:]



df.loc[df['most_often'].isna(),'most_often']=0

print(df.head())

scaler = MinMaxScaler().fit(df[features])
x = pd.DataFrame(scaler.transform(df[features]))
x.columns = features

print(x)
#colors = ['red', 'green', 'blue', 'gold']

#fig = plt.figure(figsize=(5,5))
#plt_ind=1





for i in [2,3,4,5,6,7,8,9,10]:
    print("Clusters:%d" %i)
    est = KMeans(n_clusters=i, init='random')
    est.fit(x)
    df['labels'] = est.labels_

    #centers = pd.DataFrame(est.cluster_centers_, columns=features)
    print('centers:', est.cluster_centers_)
    print('senka:', silhouette_score(df, est.labels_))

centers = pd.DataFrame(est.cluster_centers_).astype(int)

print(centers)



















"""
    #sp =fig.add_subplot(2,1,plt_ind)

    for j in range(0,i):
        cluster = df.loc[df['labels'] ==j]
        plt.scatter(cluster['height'], cluster['weight'], color = colors[j], label ="cluster %d"%j,
                    )

    sp.scatter(centers['height'], centers['weight'], color='black', marker='x', label = 'centroidi')
    plt.title('Senka %0.3f' % silhouette_score(x, est.labels_))
    plt.legend()


    plt_ind+=1

plt.tight_layout()
plt.show()

"""
