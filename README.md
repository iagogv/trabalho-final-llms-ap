# Trabalho Final — Disciplina de LLLMs e AP

Este repositório contém o **trabalho final da disciplina de LLLMs e AP**, com foco em predição de propensão de clientes bancários adquirirem produtos no mês seguinte.

## Objetivo do trabalho

Desenvolver e comparar abordagens para prever 4 targets binários de produtos bancários usando:

- modelos clássicos de Machine Learning;
- Transformer treinado do zero;
- Transformer com pré-treinamento auto-supervisionado;
- abordagem híbrida combinando embeddings do Transformer com features agregadas.

Dataset utilizado: **ai-lab/MBD-mini** (subset multimodal bancário, via Hugging Face).
Métrica principal: **ROC-AUC** por target e média geral.

## O que foi feito

1. **Setup e ingestão dos dados**
	- Download e extração dos arquivos (`detail`, `targets`, `client_split`).
	- Leitura em Parquet e consolidação dos dataframes.

2. **Análise exploratória (EDA)**
	- Distribuição dos 4 targets.
	- Volume de transações por cliente, distribuição de valores e tipos de evento.
	- Análise temporal de transações.

3. **Pré-processamento**
	- Construção de amostras com **sliding windows** por cliente/mês.
	- Transformação de `amount` com **log1p + QuantileTransformer**.
	- Criação de features temporais cíclicas (dia da semana e dia do mês).
	- Separação por folds para treino, validação e teste.

4. **Baselines tradicionais**
	- Engenharia de features agregadas por cliente.
	- Treinamento e avaliação de:
	  - Regressão Logística
	  - Random Forest
	  - XGBoost
	  - LightGBM

5. **Modelo Transformer para transações**
	- Encoder sequencial com embeddings categóricos + features contínuas.
	- Positional encoding temporal.
	- Cabeça de classificação multi-target.

6. **Pré-treinamento auto-supervisionado (MTM)**
	- Estratégia tipo BERT com mascaramento de transações.
	- Reconstrução de atributos mascarados (categóricos + contínuos).

7. **Classificação supervisionada**
	- Comparação entre:
	  - **Fine-Tuned** (encoder pré-treinado)
	  - **Scratch** (treinado do zero)

8. **Abordagem híbrida**
	- Extração de embeddings (token CLS) do Transformer fine-tuned.
	- Combinação com features agregadas.
	- Classificação final com LR e LightGBM.

9. **Comparação final e interpretação**
	- Consolidação de resultados em tabelas e gráficos.
	- Análise de atenção do Transformer.
	- Importância de features no modelo híbrido.

## Principais resultados

Foram realizadas 4 execuções principais (arquivos em `artefatos/model_exec_*`).

Para manter consistência com o paper, os resultados abaixo são reportados como **média ± desvio padrão** nas execuções:

- **Emb Only (LR)**: **0,7862 ± 0,0064** (maior média global)
- **Transformer (Fine-Tuned, TPE)**: **0,7852 ± 0,0178**
- **Hybrid (LR)**: **0,7831 ± 0,0027**
- **Transformer (Scratch)**: **0,7674 ± 0,0056**

Assim, conforme o paper, a melhor média geral foi da abordagem **Emb Only (LR)**, e o pré-treinamento apresentou ganho em relação ao modelo treinado do zero.

## Estrutura do repositório

- `notebook/02-trabalho_final_iago_vargas.ipynb`: notebook principal com todo o pipeline.
- `artefatos/model_exec_1` a `model_exec_4`: resultados finais de execuções.
- `paper/`: materiais de apoio para o relatório/artigo.

## Autor

**Iago Garcia Vargas**
