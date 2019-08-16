import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


df = pd.read_excel('chosen123456.xlsx');
features = df.columns[1:].tolist()

#scaler = MinMaxScaler().fit(df[features]);
#x = pd.DataFrame(scaler.transform(df[features]));
#x.columns = features;
df.loc[df['most_often'].isna(),'most_often']=0
print(df.head())
"""
hijerarhijsko klasterovanje:
AgglomerativeClustering()   
"""
est = AgglomerativeClustering(n_clusters=6, linkage='complete', affinity='manhattan');
est.fit_predict(df);
df['labels'] = est.labels_;

print(df);


print(est.children_); # ovo children ne znam sta je ?
print(est.n_connected_components_);

