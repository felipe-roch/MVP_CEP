# Previsão de Default em Cartão de Crédito
## Suporte à Tomada de Decisão Estratégica em Gestão de Risco

**Autor:** Felipe da Rocha Ferreira - 211038342
**Disciplina:** Controle Estatístico de Processos (CEP)
**Professor:** André Luiz Marques Serrano

---

## Resumo Executivo

Este projeto propõe o desenvolvimento de um modelo de aprendizado de máquina supervisionado para prever a inadimplência de clientes de cartão de crédito, com base em perfil demográfico e histórico de pagamentos. O problema é enquadrado como classificação binária e visa apoiar decisões estratégicas de gestão de risco em instituições financeiras, como concessão de crédito, definição de limites e acionamento de políticas de cobrança preventiva.

---

## 📋 Problema de Negócio

O risco de crédito é um dos pilares centrais da gestão estratégica em instituições financeiras. Quando um cliente de cartão de crédito deixa de pagar sua fatura (default), a instituição emissora incorre em perdas financeiras diretas e enfrenta desafios regulatórios relacionados à manutenção de capital mínimo.

**Objetivo:** Desenvolver um modelo preditivo que:
- Classifique clientes com alta probabilidade de inadimplência no mês seguinte
- Reduza falsos negativos (inadimplentes não detectados — maior risco financeiro)
- Minimize falsos positivos (bons pagadores bloqueados — custo de oportunidade)
- Seja interpretável e operacionalizável por equipes de risco de crédito

---

## 📊 Dataset

**Fonte:** UCI Machine Learning Repository / [Default of Credit Card Clients](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)

**Características:**
- **Número de amostras:** 30.000 clientes de cartão de crédito
- **Número de features:** 24 variáveis preditoras
- **Variável-alvo:** `default.payment.next.month` (0 = não inadimpliu, 1 = inadimpliu)
- **Proporção de default:** ~22% (dataset desbalanceado)
- **Período:** Abril a setembro de 2005 (Taiwan)
- **Licença:** CC0 Public Domain

**Grupos de Variáveis:**

| Grupo | Variáveis | Descrição |
|-------|-----------|-----------|
| Perfil do cliente | LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE | Dados demográficos e limite concedido |
| Histórico de pagamento | PAY_0 a PAY_6 | Status mensal: -1=pago, 1–9=meses de atraso |
| Faturas e pagamentos | BILL_AMT1–6, PAY_AMT1–6 | Valores de fatura e pagamento dos últimos 6 meses |

---

## 🔬 Metodologia

### Etapa 1: Definição do Problema

**Premissas e Hipóteses:**
- O histórico recente de atrasos (últimos 3 a 6 meses) é o preditor mais forte de default futuro
- O nível de utilização do limite de crédito tem relação positiva com a probabilidade de inadimplência
- Variáveis demográficas possuem poder preditivo secundário, relevante quando combinadas com variáveis financeiras
- Modelos baseados em árvores devem superar modelos lineares, dada a não-linearidade esperada
- O desbalanceamento de classes (~22% de default) exige métricas além da acurácia simples

**Restrições de Seleção:**
- Todos os 30.000 registros serão utilizados, sem filtros adicionais
- Variáveis identificadoras (ID) serão removidas antes da modelagem
- Variáveis categóricas serão codificadas adequadamente
- Desbalanceamento tratado via `class_weight` ou SMOTE conforme análise exploratória

### Etapa 2: Preparação de Dados

**Tratamento planejado:**
- Remoção de variáveis sem valor preditivo (ID do cliente)
- Encoding de variáveis categóricas (sexo, escolaridade, estado civil)
- Normalização das variáveis numéricas contínuas (StandardScaler)
- Tratamento do desbalanceamento de classes

**Divisão dos Dados:**
- Treino: 70%
- Validação: 15%
- Teste: 15%
- Seed fixada em 42 para reprodutibilidade
- Validação cruzada estratificada (5-fold)

### Etapa 3: Modelagem Preditiva

**Algoritmos Selecionados:**

1. **Regressão Logística** — baseline interpretável; produz probabilidades calibradas diretamente
2. **Árvore de Decisão** — interpretação visual das regras de decisão; identifica interações simples
3. **Random Forest** — ensemble não-linear; reduz overfitting; robusto ao desbalanceamento com `class_weight`
4. **XGBoost** — estado da arte para dados tabulares; permite otimização avançada de hiperparâmetros

### Etapa 4: Otimização de Hiperparâmetros

**Modelos a otimizar:** Random Forest e XGBoost

**Estratégia:** GridSearchCV ou RandomizedSearchCV

```python
# Exemplo de grid para Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'class_weight': ['balanced', None]
}
```

### Etapa 5: Avaliação e Validação

**Métricas de Avaliação:**

| Métrica | Papel | Justificativa |
|---------|-------|---------------|
| AUC-ROC | Principal | Independente do limiar; ideal para datasets desbalanceados |
| F1-Score | Complementar | Equilibra precisão e recall |
| Precisão | Complementar | Taxa de acerto entre os classificados como inadimplentes |
| Recall | Complementar | Proporção de inadimplentes reais detectados |

---

## 🛠️ Requisitos Técnicos

### Dependências
```
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
matplotlib
seaborn
kagglehub
```

### Ambiente
- Python 3.9+
- Sem requisitos computacionais elevados (~1GB RAM)

---

## 📁 Estrutura do Projeto

```
MVP_CEP/
├── default_credit_card.ipynb   # Notebook principal
└── README.md
```

---

## ⚠️ Limitações e Considerações

1. **Dataset:**
   - Dados de 2005, podendo não refletir padrões de crédito atuais
   - Desbalanceamento entre classes (~22% default)
   - Ausência de variáveis macroeconômicas contextuais

2. **Modelo:**
   - Não captura dependências temporais entre os meses
   - Requer retreinamento periódico com dados atualizados
   - Limiar de decisão deve ser calibrado conforme política de risco da instituição

3. **Recomendações futuras:**
   - Incorporar variáveis macroeconômicas (taxa de juros, inflação)
   - Explorar modelos sequenciais (LSTM) para capturar evolução temporal dos pagamentos
   - Validar externamente com dados de outra instituição ou período

---

## 📚 Referências

### Controle Estatístico de Processos
- Montgomery, D. C. (2020). *Introduction to Statistical Quality Control*. John Wiley & Sons.

### Machine Learning
- Scikit-learn Documentation: https://scikit-learn.org/
- Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. O'Reilly.

### Dataset
- UCI Machine Learning Repository — Default of Credit Card Clients:
  https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- Kaggle:
  https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset

---

**Última atualização:** Abril de 2026
**Status:** Em desenvolvimento