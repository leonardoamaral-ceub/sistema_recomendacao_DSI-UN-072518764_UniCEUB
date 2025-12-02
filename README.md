# Sistema de Recomendação Híbrido de Livros (Goodreads)

Este projeto implementa um sistema de recomendação de livros utilizando um modelo **híbrido em cascata**, combinando:

- **Filtragem Colaborativa baseada em usuários (CF)**  
- **Filtragem Baseada em Conteúdo (CB)**, a partir das tags dos livros

A solução é exposta por meio de uma **API em FastAPI**, que pode ser executada em ambiente local (Python 3.11) ou em **Docker**.

---

## Objetivo

Recomendar livros para um usuário a partir:

- Do histórico de avaliações (notas) de usuários  
- Das características de conteúdo (tags do Goodreads)  
- Da combinação das duas abordagens em um modelo híbrido

Ao final do fluxo de recomendação, o sistema retorna **5 livros**:

- **2 livros** provenientes da filtragem colaborativa (CF)  
- **3 livros** provenientes da filtragem por conteúdo (CB), similares aos 2 primeiros

---

## Base de Dados

Os dados foram obtidos do conjunto público do **Goodreads** no Kaggle:

> https://www.kaggle.com/code/philippsp/book-recommender-collaborative-filtering-shiny/input

Arquivos utilizados:

- `books.csv` — informações dos livros (`book_id`, `original_title`, etc.)  
- `ratings.csv` — avaliações dos usuários (`user_id`, `book_id`, `rating`)  
- `tags.csv` — descrição das tags  
- `book_tags.csv` — associação entre livros e tags

Esses arquivos devem estar disponíveis no caminho esperado pelo código (por exemplo, em uma pasta `data/`).

---

## Metodologia de Recomendação

### 1. Filtragem Colaborativa (CF)

Implementada com a biblioteca **Surprise** e o algoritmo `KNNBaseline`:

- Usa pares (`user_id`, `book_id`, `rating`)  
- Treina um modelo com similaridade `pearson_baseline`  
- Para um usuário, estima as notas de livros não avaliados  
- Seleciona os **2 livros com maior nota prevista** como recomendação inicial

### 2. Filtragem Baseada em Conteúdo (CB)

Implementada com **TF-IDF** + **similaridade do cosseno**:

1. Une `tags` e `book_tags`, gerando um texto de tags (`all_tags`) para cada livro  
2. Cria uma matriz TF-IDF com `TfidfVectorizer`  
3. Calcula a matriz de similaridade de cosseno entre todos os livros  
4. A partir dos 2 livros da CF, encontra livros similares em conteúdo, excluindo:  
   - Livros já avaliados pelo usuário  
   - Livros já recomendados na etapa colaborativa  

Seleciona **3 livros** com maior similaridade.

### 3. Recomendação Híbrida em Cascata

Fluxo:

1. CF escolhe **2 livros “sementes”** para o usuário  
2. CB encontra **3 livros similares** a esses 2  
3. Junta os resultados em uma lista de **5 recomendações** (2 CF + 3 CB)

---

## Tecnologias Utilizadas

- **Python 3.11**  
- **FastAPI** — API REST  
- **Uvicorn** — servidor ASGI  
- **pandas**, **numpy**  
- **scikit-learn**
  - `TfidfVectorizer`
  - `cosine_similarity`
- **Surprise**
  - `KNNBaseline`, `Dataset`, `Reader`
- **dill** (se utilizado para serialização de objetos)  
- **Docker** (containerização)

⚠️ Observação sobre Python / NumPy  
O **Surprise** é sensível à combinação de versões de Python e NumPy.  
Este projeto foi padronizado com **Python 3.11** e **NumPy 1.26.4**, que funcionam corretamente.  
Por isso é obrigatório rodar em um ambiente virtual com essas versões.

---

## Como Executar em Ambiente Local (sem Docker)

### 1. Criar ambiente virtual com Python 3.11

Na raiz do projeto:

    python3.11 -m venv .venv311

### 2. Ativar o ambiente virtual

Windows (PowerShell / CMD):

    .\.venv311\Scripts\activate

Linux / macOS:

    source .venv311/bin/activate

### 3. Atualizar pip e ajustar NumPy

Dentro do ambiente virtual:

    pip install --upgrade pip
    pip uninstall -y numpy
    pip install numpy==1.26.4 --no-cache-dir

### 4. Instalar as demais dependências

    pip install -r requirements.txt

### 5. Subir a API com Uvicorn

Na raiz do projeto (onde o módulo `src.main` está acessível):

    uvicorn src.main:app --reload --port 8500

A documentação estará disponível em:

- Swagger UI: http://localhost:8500/docs  

---

## Como Executar com Docker

### 1. Construir a imagem

Na raiz do projeto (onde está o `Dockerfile`):

    docker build -t sistema-recomendacao:latest .

### 2. Rodar o container

    docker run --rm -p 8500:8500 --name sistema-recomendacao sistema-recomendacao:latest

A API ficará disponível em:

- http://localhost:8500/docs  

---

## Endpoints da API

A seguir, um resumo dos principais endpoints expostos pela API (ver também em `/docs`).

### 1. GET /dataset_health — Dataset Health

- Descrição: verifica se os datasets foram carregados corretamente e retorna informações básicas.  
- Parâmetros: nenhum.  
- Uso: monitorar se a API está com dados disponíveis.

Exemplo de resposta:

    {
      "status": "ok",
      "books_count": 10000,
      "ratings_count": 981756,
      "tags_count": 34250
    }

---

### 2. POST /recomenda_livro — Recomenda Livro (Colaborativo)

- Descrição: retorna recomendações de livros utilizando apenas filtragem colaborativa para um usuário.  
- Uso: obter recomendações “puras” de CF.

Body (exemplo):

    {
      "user_id": 280,
      "n": 5
    }

Resposta (exemplo):

    {
      "user_id": 280,
      "recomendacoes": [
        "Livro 1",
        "Livro 2",
        "Livro 3",
        "Livro 4",
        "Livro 5"
      ]
    }

---

### 3. POST /top_n — Top N (Colaborativo)

- Descrição: retorna o Top N de livros recomendados pela filtragem colaborativa para um usuário.  

Body (exemplo):

    {
      "user_id": 280,
      "n": 10
    }

Resposta (exemplo):

    {
      "user_id": 280,
      "top_n": [
        "Livro 1",
        "Livro 2",
        "Livro 3"
      ]
    }

---

### 4. POST /top_n_hibrida_cascata — Top N Híbrida em Cascata

- Descrição: endpoint principal do sistema, combinando CF + CB.  
- Funcionamento:
  - Usa CF para escolher `n_cf` livros iniciais.  
  - Usa CB para escolher `n_cb` livros similares.  
  - Retorna lista final híbrida.

Body (exemplo):

    {
      "user_id": 280,
      "n_cf": 2,
      "n_cb": 3
    }

Resposta (exemplo):

    {
      "user_id": 280,
      "cf": [
        "Livro CF 1",
        "Livro CF 2"
      ],
      "cb": [
        "Livro CB 1",
        "Livro CB 2",
        "Livro CB 3"
      ],
      "hibrida": [
        "Livro CF 1",
        "Livro CF 2",
        "Livro CB 1",
        "Livro CB 2",
        "Livro CB 3"
      ]
    }

Obs.: os nomes exatos dos campos podem variar de acordo com os modelos Pydantic definidos no código. Em caso de dúvida, conferir o schema direto no Swagger em “Schemas”.

---

## Estrutura do Projeto (exemplo)

    .
    ├── src
    │   ├── main.py                 # Definição da aplicação FastAPI (app, rotas)
    │   ├── recommender.py          # Lógica de recomendação CF, CB e híbrida
    │   └── ...
    ├── data
    │   ├── books.csv
    │   ├── ratings.csv
    │   ├── tags.csv
    │   └── book_tags.csv
    ├── requirements.txt
    ├── Dockerfile
    ├── README.md
    └── ...

---



## Referências

- Dataset Goodreads (Kaggle)  
- Documentação:
  - FastAPI — https://fastapi.tiangolo.com/
  - Surprise — http://surpriselib.com/
  - scikit-learn — https://scikit-learn.org/
  - NumPy — https://numpy.org/
  - pandas — https://pandas.pydata.org/
