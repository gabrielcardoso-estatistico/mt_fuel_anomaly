# Análise — Detecção de Anomalias nos Preços da Gasolina em Mato Grosso

## 1. Objetivo

O projeto investiga se técnicas de **Machine Learning não supervisionado** conseguem identificar comportamentos incomuns nos preços semanais da gasolina em Mato Grosso. A unidade de análise é **posto × semana**, combinando preço atual, contexto municipal, histórico do próprio posto, volatilidade e sazonalidade.

Uma observação marcada como anômala deve ser entendida apenas como **estatisticamente incomum**. O resultado não é evidência de fraude, abuso de preço, cartel ou irregularidade.

## 2. Base analisada

A base é proveniente da ANP e foi filtrada para Mato Grosso e gasolina comum.

| Etapa | Observações |
|---|---:|
| Mato Grosso — todos os combustíveis | 36.184 |
| Gasolina | 9.506 |
| Base pronta para ML | 8.679 |
| Treino — 2024/2025 | 6.769 |
| Teste — 1º semestre de 2026 | 1.910 |

No período de teste foram analisados **138 postos em 7 municípios**.

A divisão temporal foi preferida a uma divisão aleatória: os modelos aprendem com 2024–2025 e avaliam observações futuras de 2026.

## 3. Feature engineering

Foram utilizadas 15 variáveis, organizadas em três grupos.

**Contexto local:** preço atual, desvio em relação à mediana municipal, percentil municipal, desvio em relação à mediana de Mato Grosso e quantidade de postos pesquisados no município.

**Histórico do posto:** variação de uma semana, variação de quatro semanas, desvio em relação à média móvel recente, volatilidade de quatro e oito semanas e z-score histórico.

**Sazonalidade:** codificação seno/cosseno para mês e semana do ano.

A principal vantagem dessa construção é evitar uma definição simplista de anomalia baseada apenas em preço absoluto.

## 4. Modelos

Foram comparados quatro detectores não supervisionados:

- Isolation Forest;
- Local Outlier Factor (LOF);
- One-Class SVM;
- Autoencoder.

Além disso, os quatro scores foram transformados em percentis empíricos e combinados em um **ensemble**.

## 5. Validação controlada

Como os dados da ANP não possuem rótulo real de anomalia, foram inseridas **66 anomalias sintéticas** no período de teste, por meio de choques artificiais de alta e queda no preço. As features foram recalculadas após a injeção.

Esse procedimento fornece um benchmark controlado para comparar a capacidade de detecção dos modelos, sem afirmar que todas as observações não alteradas sejam necessariamente normais.

## 6. Desempenho dos modelos

| Modelo | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| **Isolation Forest** | **0,997** | **0,896** | **0,959** | 0,712 | **0,817** |
| Ensemble | 0,995 | 0,837 | 0,804 | 0,561 | 0,661 |
| One-Class SVM | 0,948 | 0,257 | 0,257 | **1,000** | 0,409 |
| LOF | 0,947 | 0,251 | 0,252 | 0,985 | 0,401 |
| Autoencoder | 0,904 | 0,158 | 0,163 | 0,561 | 0,253 |

![Comparação dos modelos](assets/model_comparison.svg)

O **Isolation Forest foi o melhor modelo individual**. Ele apresentou excelente capacidade de ranqueamento e, principalmente, desempenho muito superior em Precision-Recall.

LOF e One-Class SVM tiveram recalls extremamente altos, mas precisão próxima de 25%, indicando comportamento mais agressivo e maior geração de falsos positivos no benchmark sintético.

O Autoencoder convergiu e aprendeu a estrutura histórica das features, mas o erro de reconstrução foi menos discriminativo do que o score do Isolation Forest.

## 7. Resultado do ensemble em 2026

O limiar final do ensemble foi aproximadamente **0,9867**.

Entre as **1.910 observações de 2026**, o modelo sinalizou **39 anomalias**, equivalentes a **2,04%** da amostra.

A interpretação correta é que essas observações apresentam combinações incomuns de preço, trajetória histórica, volatilidade e contexto local.

## 8. Padrões municipais

| Município | Taxa de anomalias | Score máximo | Observações |
|---|---:|---:|---:|
| **Sorriso** | **10,10%** | 0,996 | 99 |
| **Sinop** | **5,60%** | 0,999 | 250 |
| Várzea Grande | 1,79% | 0,995 | 392 |
| Alta Floresta | 1,11% | 0,992 | 180 |
| Cuiabá | 1,10% | 0,997 | 456 |
| Rondonópolis | 0,31% | 0,995 | 327 |
| Cáceres | 0,00% | 0,982 | 206 |

![Taxa de anomalias por município](assets/municipality_anomaly_rates.svg)

Sorriso e Sinop apresentaram maior concentração de flags no recorte de 2026. Esse resultado não deve ser usado como ranking de qualidade do mercado ou de conduta dos postos, pois a cobertura da amostra é limitada e desigual entre municípios.

## 9. Estudo de caso — Sinop

A observação com maior score do ensemble apresentou aproximadamente **0,999**.

- preço: **R$ 6,99/L**;
- mediana municipal: **R$ 6,89/L**;
- diferença em relação à mediana: apenas **+1,45%**;
- z-score histórico: **2,21**;
- percentil do Isolation Forest: **0,9999**;
- percentil do LOF: **0,9973**;
- percentil do One-Class SVM: **0,9996**;
- percentil do Autoencoder: **0,9994**.

![Estudo de caso](assets/case_study.svg)

Esse é um dos principais achados do projeto: **um preço pode ser anômalo sem ser muito diferente da mediana atual do município**.

O modelo identificou uma combinação incomum de trajetória histórica e comportamento multivariado. Em uma semana anterior, o mesmo posto registrou R$ 7,09/L após uma alta semanal de aproximadamente **18,36%**, acompanhada de z-score histórico extremo.

## 10. Interpretabilidade

A análise SHAP aplicada ao score do Isolation Forest indicou maior influência de variáveis associadas a:

1. preço atual;
2. variação semanal;
3. diferença para a média móvel recente;
4. volatilidade de curto prazo;
5. z-score histórico;
6. volatilidade de prazo mais longo;
7. número de postos pesquisados no município;
8. desvio em relação ao preço municipal.

Os valores SHAP explicam o comportamento do modelo e **não devem ser interpretados como efeitos causais** sobre os preços.

## 11. Conclusões

O resultado mais relevante é que a detecção de anomalias se beneficia fortemente do contexto temporal e local. Um limite fixo de preço deixaria de identificar comportamentos incomuns que são visíveis quando o histórico do próprio posto é incorporado.

O Isolation Forest apresentou o melhor equilíbrio entre sensibilidade e precisão no benchmark sintético. O ensemble manteve excelente poder de ranqueamento, mas foi mais conservador no limiar utilizado.

O projeto demonstra uma aplicação prática de Machine Learning não supervisionado em dados públicos, combinando séries temporais, feature engineering, métodos clássicos e neurais, avaliação controlada e análise geográfica.

## 12. Limitações

- apenas sete municípios de Mato Grosso estão representados no recorte de gasolina utilizado;
- não existem rótulos reais confirmando quais observações são anomalias;
- o limiar de 2% é uma escolha operacional;
- a análise não é causal;
- câmbio, petróleo, impostos, frete e logística não foram incluídos;
- flags em nível de posto não representam acusações ou indícios de ilegalidade.

## 13. Próximas extensões

- incluir etanol e diesel;
- incorporar variáveis macroeconômicas e de petróleo;
- criar detectores específicos por município;
- otimizar pesos do ensemble;
- realizar retreinamento temporal rolling;
- construir dashboard em Streamlit;
- automatizar atualização semanal com dados da ANP.
