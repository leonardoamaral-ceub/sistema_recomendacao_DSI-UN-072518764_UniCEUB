# Sistema de Recomendação Híbrido de Livros (Goodreads)

Este projeto implementa um sistema de recomendação de livros utilizando um modelo **híbrido em cascata**, combinando **filtragem colaborativa baseada em usuários** e **filtragem baseada em conteúdo**.

A aplicação expõe uma **API em FastAPI**, que pode ser executada tanto em **ambiente local (com ambiente virtual Python 3.11)** quanto em **Docker**.

---

## Objetivo

Desenvolver um sistema capaz de recomendar livros a partir:

- Das avaliações de usuários (notas atribuídas aos livros);
- Das características de conteúdo dos livros (tags do Goodreads);
- Da combinação dessas duas abordagens em um modelo híbrido.

Ao final do fluxo de recomendação, o sistema retorna **5 livros recomendados** para um usuário:

- **2 livros** sugeridos por **filtragem colaborativa (CF)**;
- **3 livros** sugeridos por **filtragem por conteúdo (CB)**, com base nesses 2 livros iniciais.

---

## Base de Dados

Os dados utilizados foram retirados do conjunto público do **Goodreads** no Kaggle:

> https://www.kaggle.com/code/philippsp/book-recommender-collaborative-filtering-shiny/input

Arquivos utilizados:

- `books.csv` — informações dos livros (ex.: `book_id`, `original_title`, etc.);
- `ratings.csv` — avaliações dos usuários (`user_id`, `book_id`, `rating`);
- `tags.csv` — descrição das tags;
- `book_tags.csv` — associação entre livros e tags.

> **Importante:** estes arquivos devem estar disponíveis no caminho esperado pelo código (por exemplo, na pasta de dados configurada na API / notebook).

---

## Metodologia de Recomendação

### 1. Filtragem Colaborativa (CF)

A primeira etapa usa **filtragem colaborativa baseada em usuários** com o algoritmo `KNNBaseline` da biblioteca **Surprise**:

- Leitura das avaliações (`user_id`, `book_id`, `rating`);
- Criação do `trainset` e treinamento do modelo `KNNBaseline` com similaridade `pearson_baseline`;
- Para um determinado usuário, o modelo estima as notas dos livros ainda não avaliados;
- São selecionados os **2 livros com maior nota prevista** como recomendações iniciais.

### 2. Filtragem Baseada em Conteúdo (CB)

A segunda etapa utiliza **filtragem baseada em conteúdo**, a partir das tags dos livros:

1. As tabelas `tags` e `book_tags` são combinadas para criar, para cada livro, um texto representando suas tags (`all_tags`);
2. É construída uma matriz **TF-IDF** dessas tags com `TfidfVectorizer`;
3. A similaridade entre livros é calculada com **similaridade do cosseno** (`cosine_similarity`);
4. A partir dos 2 livros recomendados pela CF, o sistema procura livros **similares em conteúdo**, excluindo:
   - Livros já avaliados pelo usuário;
   - Livros já recomendados pela etapa colaborativa.

Assim, são escolhidos **3 novos livros** com maior similaridade de conteúdo.

### 3. Recomendação Híbrida em Cascata

A lógica híbrida combina as duas abordagens:

1. A CF escolhe **2 livros “sementes”** com base no histórico do usuário;
2. A CB encontra **3 livros similares** a esses 2, com base nas tags;
3. O resultado final é uma lista de **5 livros recomendados** (2 CF + 3 CB).

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
- **dill** (se utilizado para serialização)
- **Docker** (containerização)

> ⚠️ **Observação importante sobre versão de Python / NumPy**  
> A biblioteca **Surprise** apresenta problemas com algumas combinações de Python e NumPy. Neste projeto, foi padronizado o uso de **Python 3.11** e **NumPy 1.26.4**, que funcionam corretamente com o Surprise.  
> Por isso é **obrigatório** rodar o projeto em um **ambiente virtual** com Python 3.11 e NumPy travado na versão `1.26.4`.

---

## Como Executar em Ambiente Local (sem Docker)

### 1. Criar ambiente virtual com Python 3.11

Na raiz do projeto, execute:

    python3.11 -m venv .venv311

### 2. Ativar o ambiente virtual

No **Windows (PowerShell / CMD)**:

    .\.venv311\Scripts\activate

No **Linux / macOS**:

    source .venv311/bin/activate

Você deve ver algo como `(.venv311)` no início da linha de comando após a ativação.

### 3. Atualizar `pip` e ajustar NumPy

Dentro do ambiente virtual, execute:

    pip install --upgrade pip
    pip uninstall -y numpy
    pip install numpy==1.26.4 --no-cache-dir

### 4. Instalar as demais dependências

Ainda dentro do ambiente virtual:

    pip install -r requirements.txt

> O arquivo `requirements.txt` contém as bibliotecas necessárias para rodar o sistema de recomendação e a API FastAPI.

### 5. Subir a API com Uvicorn

Na raiz do projeto (ou na pasta onde está o módulo `src.main`), execute:

    uvicorn src.main:app --reload --port 8500

- `src.main:app` → caminho do módulo e nome da instância do FastAPI.  
- `--reload` → recarrega automaticamente ao salvar mudanças (útil em desenvolvimento).  
- `--port 8500` → expõe a API na porta 8500.

Após esse comando, a API ficará disponível em:

- Documentação interativa (Swagger): **http://localhost:8500/docs**
- Versão alternativa (ReDoc): **http://localhost:8500/redoc** (se habilitada)

---

## Como Executar com Docker

### 1. Construir a imagem Docker

Na raiz do projeto (onde está o `Dockerfile`), execute:

    docker build -t sistema-recomendacao:latest .

- `-t sistema-recomendacao:latest` → dá o nome e a tag para a imagem.  
- `.` → indica que o build usa o `Dockerfile` do diretório atual.

### 2. Executar o container

    docker run --rm -p 8500:8500 --name sistema-recomendacao sistema-recomendacao:latest

Explicando os parâmetros:

- `--rm` → remove o container automaticamente ao parar;  
- `-p 8500:8500` → mapeia a porta 8500 do container para a 8500 da máquina host;  
- `--name sistema-recomendacao` → nome amigável para o container;  
- `sistema-recomendacao:latest` → imagem criada no passo anterior.

Com o container rodando, a API estará disponível em:

- **http://localhost:8500/docs**

---

## Estrutura do Projeto (exemplo)

    .
    ├── src
    │   ├── main.py                 # Definição da aplicação FastAPI (app)
    │   ├── recommender.py          # Lógica de recomendação híbrida (CF + CB)
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

> A estrutura pode variar levemente, mas é importante que o caminho usado em `src.main:app` seja válido e que a API consiga acessar os arquivos de dados conforme configurado no código.

---

## Próximos Passos (opcionais)

- Adicionar testes automatizados (unitários e/ou de integração) para a API;
- Criar scripts de inicialização de dados (ex.: pré-processar matrizes TF-IDF e modelos de CF);
- Configurar logs e monitoramento básico;
- Publicar a imagem em um registry (por exemplo, Docker Hub).

---

## Referências

- Conjunto de dados do Goodreads (Kaggle)  
- Documentação das bibliotecas:
  - FastAPI
  - Surprise
  - scikit-learn
  - NumPy / pandas
