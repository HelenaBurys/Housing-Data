## Este arquivo lê o modelo treinado e gera o relatório interpretável dos bairros.

import pickle
import pandas as pd
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Dicionário com descrição das variáveis (fonte: Kaggle)
FEATURE_DESCRIPTIONS = {
    'CRIM': 'Taxa per capita de crime por cidade',
    'ZN': 'Proporção de terra residencial para lotes > 25.000 sq.ft.',
    'INDUS': 'Proporção de acres comerciais não-varejistas',
    'CHAS': 'Dummy: 1 se limita com rio Charles, 0 caso contrário',
    'NOX': 'Concentração de óxidos de nitrogênio (partes por 10 milhões)',
    'RM': 'Número médio de cômodos por residência',
    'AGE': 'Proporção de unidades ocupadas construídas antes de 1940',
    'DIS': 'Média ponderada das distâncias a 5 centros de emprego',
    'RAD': 'Índice de acessibilidade a rodovias radiais',
    'TAX': 'Taxa de imposto predial por $10.000',
    'PTRATIO': 'Razão aluno-professor por cidade',
    'B': '1000(Bk - 0.63)² onde Bk = proporção de residentes negros',
    'LSTAT': 'porcentagem da população com status socioeconômico inferior'
}

# Abrir o modelo ja treinado la em "housing_cluster_treinamento.py""
cluster_housing = pickle.load(open('cluster_HousingData.pkl', 'rb'))
metadata = pickle.load(open('housing_metadata.pkl', 'rb'))

features = metadata['features']
n_clusters = metadata['n_clusters']

# Converter os centróides em DataFrame
centroides_norm = pd.DataFrame(
    cluster_housing.cluster_centers_, columns=features)

print("="*70)
print("CENTRÓIDES NORMALIZADOS (escala 0-1)")
print("="*70)
print(centroides_norm.round(3))

# Desnormalizar as colunas numéricas
## Carregar o normalizador salvo durante o treinamento
normalizador = pickle.load(open('normalizador_housing.pkl','rb'))
dados_num_desnorm = normalizador.inverse_transform(centroides_norm)

# Converter os centroides em DataFrame - Tem que criar um novo dataframe, por que?  Para ajustar os nomes das colunas
centroides_originais = pd.DataFrame(dados_num_desnorm, columns=features)

print("\n" + "="*70)
print("CENTRÓIDES DESNORMALIZADOS (valores originais)")
print("="*70)
print(centroides_originais.round(2))

# Gerar relatório descritivo de cada cluster
print("\n" + "="*70)
print("INTERPRETAÇÃO DOS SEGMENTOS")
print("="*70)

# Carregar dados originais para calcular médias globais (para comparação)
dados_orig = pd.read_csv('HousingData.csv', sep=',')
dados_orig = dados_orig.replace('NA', np.nan).astype(float)
medias_globais = dados_orig[features].mean()

for cluster_id in range(n_clusters):
    print(f"\n[TORRENT] CLUSTER {cluster_id}")
    print(f"   Tamanho: {(metadata['labels']==cluster_id).sum()} imóveis")
    print(f"   Características distintivas (vs. média global):")
    
    centroide = centroides_originais.iloc[cluster_id]
    
    for feat in features:
        diff_pct = (centroide[feat] - medias_globais[feat]) / abs(medias_globais[feat]) * 100
        if abs(diff_pct) > 25:
            direction = "ACIMA" if diff_pct > 0 else "ABAIXO"
            print(f"   - {FEATURE_DESCRIPTIONS[feat]}: {direction} ({diff_pct:+.1f}%)")
    
    # Interpretação com tags em texto simples
    if centroide['RM'] > medias_globais['RM'] and centroide['LSTAT'] < medias_globais['LSTAT']:
        print("   [NOBRE] Perfil: Bairros de alto padrao (mais comodos, menor status inferior)")
    elif centroide['RM'] < medias_globais['RM'] and centroide['LSTAT'] > medias_globais['LSTAT']:
        print("   [ECONOMICO] Perfil: Bairros com menor poder aquisitivo")
    elif centroide['CRIM'] > medias_globais['CRIM'] * 2:
        print("   [ALERTA] Perfil: Areas com taxa de criminalidade elevada")

print("\n" + "="*70)