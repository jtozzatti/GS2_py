# Prevenção de Desastres Naturais — Incêndios Florestais
### Global Solutions 2026 — FIAP | Python Avançado Aplicado à Nova Economia Espacial

![Python](https://img.shields.io/badge/Python-3.14-blue?style=flat-square&logo=python)
![NASA FIRMS](https://img.shields.io/badge/Dados-NASA%20FIRMS-red?style=flat-square)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange?style=flat-square)
![Status](https://img.shields.io/badge/Status-Concluído-green?style=flat-square)

---

##  Integrantes

| Nome | RM |
|------|----|
| João Victor Tozzatti Matiro | RM 567510 |
| Pedro Diagro Lopes | RM 568393 |

---

##  Descrição do Problema

Incêndios florestais representam uma das maiores ameaças ambientais e socioeconômicas do Brasil. Em 2024, o país registrou centenas de milhares de focos de calor detectados por satélites da NASA, causando destruição de biomas, emissões massivas de carbono e prejuízos à biodiversidade e às populações locais.

Este projeto desenvolve uma **solução inteligente baseada em dados satelitais reais** (NASA FIRMS) capaz de:

- Identificar padrões de risco de incêndio por região e período
- Classificar focos de calor por nível de risco (Baixo / Médio / Alto)
- Agrupar regiões geográficas por perfil de intensidade de fogo
- Gerar insights estratégicos para apoio à tomada de decisão de órgãos como INPE e IBAMA

---

## 🛰️ Contexto Espacial

Os dados utilizados são provenientes do **NASA FIRMS (Fire Information for Resource Management System)**, sistema que utiliza o sensor **MODIS** embarcado nos satélites Terra e Aqua para detectar focos de calor em tempo quase real, com cobertura global.

Variáveis capturadas pelo satélite:
- `brightness` — Temperatura de brilho do foco (Kelvin)
- `frp` — Fire Radiative Power: potência radiativa do fogo (MW)
- `bright_t31` — Temperatura de brilho na banda 31
- `confidence` — Confiabilidade da detecção (%)
- `acq_date / acq_time` — Data e hora de aquisição pelo satélite
- `latitude / longitude` — Coordenadas geográficas do foco

---

## Estrutura do Projeto

```
gs-incendios-florestais/
│
├── data/
│   ├── fires_raw.csv          # Dataset original NASA FIRMS
│   ├── fires_clean.csv        # Dataset tratado e com features
│   ├── graficos/              # 14 gráficos gerados
│   └── models/                # Modelos treinados (.pkl)
│
├── src/
│   ├── data_prep.py           # Limpeza, encoding, normalização, feature engineering
│   ├── eda.py                 # Análise exploratória (7 gráficos)
│   ├── model.py               # Random Forest + K-Means
│   └── avaliacao.py           # Métricas, insights e dashboard
│
├── main.py                    # Pipeline completo end-to-end
├── requirements.txt           # Dependências do projeto
└── README.md
```

---

##  Instalação e Execução

### 1. Clone o repositório
```bash
git clone https://github.com/seu-usuario/gs-incendios-florestais.git
cd gs-incendios-florestais
```

### 2. Instale as dependências
```bash
pip install -r requirements.txt
```

### 3. Adicione o dataset
Baixe o CSV do NASA FIRMS em [firms.modaps.eosdis.nasa.gov](https://firms.modaps.eosdis.nasa.gov/download/) e salve como `data/fires_raw.csv`.

### 4. Execute o pipeline completo
```bash
python main.py
```

Ou execute cada etapa individualmente:
```bash
python src/data_prep.py   # Etapa 1: Preparação
python src/eda.py         # Etapa 2: EDA
python src/model.py       # Etapa 3: Modelos
python src/avaliacao.py   # Etapa 4: Avaliação
```

---

##  Etapas do Projeto

### 1. Preparação dos Dados (`data_prep.py`)
- Carregamento do CSV NASA FIRMS (527.894 registros)
- Remoção de duplicatas e tratamento de valores nulos
- **Feature engineering:**
  - `mes`, `ano`, `dia_semana` — extraídos de `acq_date`
  - `estacao` — estação do ano (hemisfério sul)
  - `periodo_dia` — Manhã / Tarde / Noite / Madrugada
  - `nivel_risco` — classificação por brightness/FRP (Baixo/Médio/Alto)
- Label Encoding de variáveis categóricas
- Normalização Min-Max de variáveis numéricas

### 2. Análise Exploratória (`eda.py`)
7 gráficos gerados em `data/graficos/`:

| # | Gráfico | Tipo |
|---|---------|------|
| 01 | Focos de incêndio por mês | Barplot |
| 02 | Distribuição de Brightness | Histograma + KDE |
| 03 | Correlação entre variáveis | Heatmap |
| 04 | FRP por nível de risco | Boxplot |
| 05 | Focos por estação do ano | Barplot |
| 06 | Série temporal por mês | Lineplot |
| 07 | Mapa geográfico de focos | Scatter geográfico |

### 3. Modelagem Inteligente (`model.py`)

#### Random Forest Classifier
- **Objetivo:** Prever o nível de risco de cada foco (Baixo / Médio / Alto)
- **Features:** brightness, FRP, bright_t31, confidence, scan, track, mês, hora, estação
- **Split:** 80% treino / 20% teste
- **Parâmetros:** 100 árvores, profundidade máxima 12

#### K-Means Clustering
- **Objetivo:** Agrupar regiões geográficas por perfil de intensidade
- **Features:** latitude, longitude, brightness, FRP
- **K escolhido:** 3 (validado pelo Elbow Method)

### 4. Avaliação e Insights (`avaliacao.py`)
- Métricas completas de classificação e clustering
- Dashboard visual com painéis de insights
- Análise estratégica de impacto

---

##  Resultados

### Random Forest Classifier
| Métrica | Valor |
|---------|-------|
| Accuracy | ~95%+ |
| Precision | ~0.95+ |
| Recall | ~0.95+ |
| F1-Score | ~0.95+ |

### K-Means Clustering
| Métrica | Valor |
|---------|-------|
| K (clusters) | 3 |
| Silhouette Score | > 0.50 |

---

##  Insights Estratégicos

- **Sazonalidade crítica:** os meses de agosto, setembro e outubro concentram a maior parte dos focos, coincidindo com o período seco no Cerrado e Amazônia
- **Estação crítica:** o Inverno (hemisfério sul) é a estação com maior incidência de focos de alta intensidade
- **Zonas de risco:** o clustering identifica 3 perfis geográficos distintos, permitindo priorização de recursos por região
- **Aplicação prática:** o modelo pode ser integrado a sistemas de alerta automático, reduzindo o tempo de resposta de brigadistas e otimizando a alocação de aeronaves e equipes terrestres

---

##  Tecnologias Utilizadas

| Biblioteca | Uso |
|------------|-----|
| `pandas` | Manipulação de dados |
| `numpy` | Operações numéricas |
| `scikit-learn` | Machine Learning |
| `matplotlib` | Visualizações |
| `seaborn` | Gráficos estatísticos |
| `scipy` | KDE e estatística |
| `joblib` | Serialização de modelos |

---

## Fonte dos Dados

**NASA FIRMS — Fire Information for Resource Management System**
- Sensor: MODIS C6.1
- Satélites: Terra e Aqua
- Região: Brasil
- Período: 2024
- Link: [firms.modaps.eosdis.nasa.gov](https://firms.modaps.eosdis.nasa.gov)

---

*FIAP — Global Solutions 2026 | Python Avançado Aplicado à Nova Economia Espacial*
