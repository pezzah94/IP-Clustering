import pandas as pd
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# ilustracija DBSCAN algoritma
df = pd.read_excel('chosen123456.xlsx')
features = df.columns.tolist();

x_original = df[features]

df.loc[df['role'].isna(),'role']=0
df.loc[df['industry'].isna(),'industry']=0
df.loc[df['most_often'].isna(),'most_often']=0
df.loc[df['edu'].isna(),'edu']=0

f = ['industry', 'most_often']
x_original = df[f]

print(x_original)


# normalizaciju podataka, treba da pitam zasto se ovo radi?
scaler = MinMaxScaler().fit(x_original)
x = pd.DataFrame(scaler.transform(x_original))

print(x)

#kako radi DBSCAN? koji su parametri?

#eps - rastojanje centralne tacke, oko koje se opisuje "krug"
#min_samples - minimalan broj tacaka u tom radijusu neophodnim da se tacka proglasi za centralnu
#metric -
fig = plt.figure(figsize=(5,5));
plt_ind=1;

#parametri po defaultu eps = 0.5, min_samples = 5, metric = 'euclidean'
est = DBSCAN(eps=1); #zasto se ovo zove est?
est.fit_predict(x) #prosledimo podatke za klasterovanje

print(est.labels_)

#x_original.loc['labels'] = est.labels_; #dodas pored oznake klastera


print(x_original)
colors = ['green', 'orange'];



for j in range(-1, 1):
    #cluster = x_original.loc[x_original['labels'] == j];
    plt.scatter(x_original['industry'], x_original['most_often'],  color=colors[j],
                label="cluster %d" % j);  # ucrtavanje tacaka na plot



plt.tight_layout();
plt.show();


print(est.core_sample_indices_); #indeksi elemenata koji pripadaju klasteru
print(est.labels_) #oznake klastera

