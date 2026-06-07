# LLM Local RAG Pipeline

Este projeto implementa um pipeline completo de **Retrieval-Augmented Generation (RAG)** focado em privacidade de dados e executado 100% em ambiente local. A arquitetura foi desenhada de forma híbrida: utiliza o **LlamaIndex** para o parsing semântico inteligente e o **LangChain** para a estruturação de chains, busca por limiar de similaridade e integração com o banco vetorial **FAISS**.

O motor de inferência (LLM e Embeddings) é gerenciado localmente através do **Ollama**.

---

## Destaques da Arquitetura Técnica

O grande diferencial deste projeto está no design modular e na escolha das estratégias de processamento de dados:

### 1. Ingestão Dinâmica e *Semantic Chunking*
Em vez de quebrar o texto por número fixo de caracteres (o que corrompe o sentido de frases e parágrafos), o pipeline utiliza o **`SemanticSplitterNodeParser` do LlamaIndex**. 
* O algoritmo calcula o embedding das sentenças em tempo real e encontra os pontos de ruptura ideais com base na mudança temática (threshold de 85% de similaridade).
* Isso garante que os fragmentos (*chunks*) enviados para o banco vetorial preservem contextos lógicos completos.

### 2. Pipeline Híbrido (LlamaIndex ➔ LangChain)
O projeto demonstra proficiência na integração de ecossistemas de IA ao extrair dados via LlamaIndex e convertê-los dinamicamente em documentos estruturados nativos do LangChain (`LCDocument`). Essa abordagem permite aproveitar o melhor motor de indexação (LlamaIndex) e a melhor infraestrutura de chains (`|` - LCEL) do LangChain.

### 3. Recuperação por Limiar de Similaridade (*Score Threshold*)
A busca na base vetorial do **FAISS** não utiliza apenas o clássico "Top-K" cego. Foi implementado o filtro por `similarity_score_threshold` com limiar de 0.5. Se nenhum documento atingir o nível mínimo de relevância matemática com a pergunta, o pipeline barra o envio de informações irrelevantes para o contexto, reduzindo drasticamente as chances de alucinação da LLM.

### 4. Estratégia de Dados (*Small Data*)
A ingestão foi aplicada intencionalmente nas páginas iniciais do documento de teste. Arquivos brutos e não tratados (como artigos da Wikipedia) possuem excesso de metadados, tabelas e links externos que geram ruído nos embeddings. A limitação controlada de páginas permitiu calibrar com precisão o algoritmo de split semântico e os filtros de busca antes de escalar o pipeline.

---

## Stack Tecnológica & Modelos

* **Orquestração & Parsing:** LlamaIndex & LangChain
* **Banco Vetorial:** FAISS (Facebook AI Similarity Search)
* **Leitor de PDF:** PDFPlumber
* **Ambiente de Modelos:** Ollama (Execução Local)
* **Embedding Model:** `mxbai-embed-large:latest`
* **LLM Local:** `qwen3:4b-instruct`

---

## Como Testar e Usar

O sistema foi unificado em uma interface de linha de comando (CLI) interativa. Ao rodar o script, um menu permite alternar entre o ciclo de criação e o de consumo da base de dados.

### 1. Instalação das Dependências
Com o terminal aberto na raiz do projeto, instale todas as bibliotecas necessárias rodando o comando:

```bash
pip install -r requirements.txt
