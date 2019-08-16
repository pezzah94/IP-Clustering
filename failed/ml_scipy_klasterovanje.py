import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib

df = pd.read_excel('chosen123456.xlsx')


features = df.columns[1:].tolist();
#df.set_index('breed', inplace=True);
#df['num'] = range(0,11)

df.loc[df['most_often'].isna(),'most_often']=0

print(df)


scaler = MinMaxScaler().fit(df[features]);
x = pd.DataFrame(scaler.transform(df[features]));
x.columns = features;
x.index = df.index
print(x)



matplotlib.rcParams['font.size']=8
#pozivamo funkciju linkage iz scipy.cluster.hierarchy

Z = linkage(x, 'single') # kako se tumaci ovo, [ i-cvor, k-cvor, dist(i,k), num_in_clust()]


print('Z je ovo:', Z, sep='\n')

fig = plt.figure(figsize=(25, 10));
fig.add_subplot(2,1,1);

dn = dendrogram(Z, labels=x.index, leaf_font_size=8, color_threshold=0.3, distance_sort=True);

plt.title('Dendogram');

fig.add_subplot(2,1,2);
colors = ['red', 'green', 'blue', 'orange', 'gold', 'm', 'black', 'brown'];

niz = fcluster(Z, t=0.3, criterion='distance');
df['label'] = niz;

print(df);

#sa ovim rangeovima, pazi sta indeksiras i kako?
#ovde se ide po klasterima
for i in range(1, max(df['label'])+1):
    cluster = df.loc[df['label']==i]
    plt.scatter(cluster['RPy'], cluster['most_often'], color=colors[i-1], label='Klaster %d' %i)

plt.legend()
print(niz);



plt.tight_layout();
plt.show();




