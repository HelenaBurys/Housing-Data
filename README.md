## Apresentação do Exercicio 

1. Utilize o arquivo HousingData.csv disponível na pasta da Aula6
2. A descrição do arquivo pode ser encontrada em https://www.kaggle.com/code/prasadperera/the-boston-housing-dataset
3. Treine um modelo de clusters conforme nossas aulas
4. Implemente um módulo de descrição dos segmentos obtidos durante o treinamento
5. Implemente um módulo de inferência que receba os dados de um imóvel desconhecido e informe a qual cluster tal imóvel pertence

### Descrição do Arquivo 

CRIM - taxa de criminalidade per capita por cidade  
ZN - proporção de terrenos residenciais zoneados para lotes acima de 25.000 pés quadrados.  
INDUS - proporção de acres de negócios não comerciais por cidade.  
CHAS - Variável fictícia do rio Charles (1 se o rio do trato delimita; 0 caso contrário)  
NOX - concentração de óxidos nítricos (partes por 10 milhões)  
RM - número médio de quartos por habitação  
IDADE - proporção de unidades ocupadas pelo proprietário construídas antes de 1940  
DIS - distâncias ponderadas para cinco centros de emprego de Boston  
RAD - índice de acessibilidade a rodovias radiais  
IMPOSTO - taxa de imposto sobre a propriedade de valor total por US$ 10.000  
PTRATIO - proporção aluno-professor por cidade  
B - 1000(Bk - 0,63)^2 onde Bk é a proporção de negros por cidade  
LSTAT - % de status mais baixo da população  
MEDV - Valor médio de casas ocupadas pelo proprietário em US$ 1.000  

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

### Etapas do meu projeto:
- Treinamento do Modelo - OK
- Interpretação dos Centroides
- Inferência / Previsão

NA - atributo categorico - moda | atributo numerico - media

Se a coluna for normal usa média se for nao normal usa mediana 
