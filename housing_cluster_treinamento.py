## Este arquivo treina o modelo, gera o gráfico do cotovelo (de 1 a 506) e salva tudo que os outros arquivos precisam.

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import pickle
import matplotlib.pyplot as plt
import math
import numpy as np

#Abrir  o arquivo CSV (dataset)
dados = pd.read_csv('HousingData.csv')

# Tratar valores NA (substituir pela média da coluna)
dados = dados.replace('NA', np.nan).astype(float)
dados = dados.fillna(dados.mean())

# Separar features para clustering (EXCLUIR MEDV - target)
# MEDV não deve ser usado no clustering pois é aprendizado não supervisionado
features_clustering = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 
                       'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

dados_num = dados[features_clustering].copy()  #Sem isso, o Python não sabia quais colunas usar

# Normalizar numéricos com MinMaxScaler
scaler=MinMaxScaler()
normalizador_housing = scaler.fit(dados_num)

# Salvar o modelo normalizador
pickle.dump(normalizador_housing, open('normalizador_housing.pkl', 'wb'))

# Normalizar os dados numéricos
dados_num_norm = normalizador_housing.fit_transform(dados_num)
dados_num_norm_df = pd.DataFrame(dados_num_norm, columns=features_clustering) # Converter o array NumPy para DataFrame temporário só para imprimir bonito no terminal

print("[OK] Dados normalizados:")
print(dados_num_norm_df.head())

# Hiperparametrizar antes do treinamento
distorcoes = []
K = range(1,506)  # Testar de 1 a 505 clusters
for i in K:
    #Treinando interativamente e aumentando o numero de clusters
    cluster_HousingData = KMeans(n_clusters=i, 
                          random_state=42).fit(dados_num_norm)
    
    #Calcular a distorção
    distorcoes.append(
        sum(
            np.min(
                cdist(
                    dados_num_norm, cluster_HousingData.cluster_centers_, 'euclidean'), 
                axis=1) / dados_num_norm.shape[0]
            )
        )   
    
# Plotar gráfico do cotovelo 
plt.figure(figsize=(8, 5))
plt.plot(K, distorcoes, 'bo-')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Distorção')
plt.title('Método do Cotovelo - Boston Housing')
plt.grid(True)
plt.savefig('elbow_housing.png', dpi=300)
plt.show()    

#Determinar o número otimo de clusters
x0 = K[0]
y0 = distorcoes[0]
xn = K[-1]
yn = distorcoes[-1]

distancias = []
for i in range(len(distorcoes)):
    x = K[i]
    y = distorcoes[i]
    numerador = abs((yn - y0) * x - (xn - x0) * y + xn * y0 - yn * x0)
    denominador = math.sqrt((yn - y0)**2 + (xn - x0)**2)
    distancias.append(numerador / denominador)
    
numero_clusters_otimo = K[distancias.index(np.max(distancias))]
print("\n[OK] Número ótimo de clusters = ", numero_clusters_otimo)

# Treinar e salvar o modelo de clusters
cluster_HousingData = KMeans(
    n_clusters=numero_clusters_otimo, random_state=42, 
    n_init=10).fit(dados_num_norm) # n_init=10: Adicionado ao KMeans para evitar warnings do scikit-learn e garantir estabilidade nos resultados

pickle.dump(cluster_HousingData, open('cluster_HousingData.pkl', 'wb'))

# Salvar metadados 
metadata = {
    'features': features_clustering,
    'n_clusters': numero_clusters_otimo,
    'labels': cluster_HousingData.labels_
}
pickle.dump(metadata, open('housing_metadata.pkl', 'wb'))

print("[OK] Modelos, normalizador e metadados salvos com sucesso!")