## Este arquivo permite testar novos imóveis (inferência).

import pickle
import pandas as pd
import numpy as np

# Carregar metadados para saber a ordem exata das colunas usadas no treino
metadata = pickle.load(open('housing_metadata.pkl', 'rb'))
features = metadata['features']

imovel_template = pd.DataFrame(columns=features)

# Valores na ordem: CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT
novo_imovel = pd.DataFrame([[0.15, 0.0, 8.0, 0, 0.55, 6.2, 70.0, 
                             3.8, 5.0, 350.0, 19.0, 390.0, 12.0]], columns=features)

# Normalizar o novo imóvel
# Carregar o normalizador salvo durante o treinamento
normalizador = pickle.load(open('normalizador_housing.pkl', 'rb'))
novo_imovel_norm = normalizador.transform(novo_imovel)
novo_imovel_norm = pd.DataFrame(novo_imovel_norm, columns=features)

# Concatenar com o template para garantir a mesma estrutura do treinamento
# e preencher com 0 qualquer coluna que possa faltar (segurança)
novo_imovel_normalizado = pd.concat([novo_imovel_norm, imovel_template]).fillna(0)

# Inferir o cluster ao qual o imóvel pertence
# Carregar o modelo de cluster salvo
cluster_housing = pickle.load(open('cluster_HousingData.pkl', 'rb'))
cluster_imovel = cluster_housing.predict(novo_imovel_normalizado)










