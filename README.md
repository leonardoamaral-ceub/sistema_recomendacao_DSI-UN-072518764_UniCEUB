# Sistema de Recomendação Híbrido de Livros (Goodreads)

Este projeto implementa um sistema de recomendação de livros utilizando um modelo **híbrido em cascata**, combinando **filtragem colaborativa baseada em usuários** e **filtragem baseada em conteúdo**.  

O trabalho foi desenvolvido como parte da atividade *“Desenvolvimento de um Sistema de Recomendação”*, cuja proposta inclui uso de **FastAPI** para criação de API e **Docker** para containerização da aplicação.:contentReference[oaicite:0]{index=0}  

---

## Objetivo

Desenvolver um sistema capaz de recomendar livros a partir:

- Das avaliações de usuários (notas atribuídas aos livros);
- Das características de conteúdo dos livros (tags do Goodreads);
- Da combinação dessas duas abordagens em um modelo híbrido.

Ao final do fluxo de recomendação, o sistema retorna **5 livros recomendados** para um usuário:

- **2 livros** sugeridos por **filtragem colaborativa**;
- **3 livros** sugeridos por **filtragem por conteúdo**, com base nesses 2 livros iniciais.

---

## Base de Dados

Os dados utilizados foram retirados do conjunto público do **Goodreads** no Kaggle:  

> https://www.kaggle.com/code/philippsp/book-recommender-collaborative-filtering-shiny/input :contentReference[oaicite:1]{index=1}  

Arquivos utilizados:

- `books.csv` — informações dos livros (ex.: `book_id`, `original_title`, etc.);
- `ratings.csv` — avaliações dos usuários (`user_id`, `book_id`, `rating`);
- `tags.csv` — descrição das tags;
- `book_tags.csv` — associação entre livros e tags.

> **Importante:** estes arquivos devem estar na **mesma pasta** do notebook `recomenda_hibrida_livros.ipynb` ou com os caminhos ajustados no código.

---

## Metodologia

### 1. Filtragem Colaborativa (CF)

A primeira etapa usa **filtragem colaborativa baseada em usuários** com o algoritmo `KNNBaseline` da biblioteca **Surprise**:

- Leitura das avaliações com `Reader` e `Dataset`;
- Criação do `trainset` com `train_test_split`;
- Treinamento do modelo `KNNBaseline` usando similaridade `pearson_baseline`;
- Para cada usuário, o modelo estima a nota que ele daria a livros ainda não avaliados;
- São selecionados os **2 livros com maior nota prevista** como recomendações iniciais.

Essa lógica é encapsulada em funções como `recomenda_livro` e `top_n`, além de ser usada dentro da função híbrida.  

---

### 2. Filtragem Baseada em Conteúdo (CB)

A segunda etapa utiliza **filtragem baseada em conteúdo**, a partir das tags dos livros:

1. As tabelas `tags` e `book_tags` são combinadas para criar, para cada livro, um texto representando suas tags (`all_tags`);
2. É construída uma matriz **TF-IDF** dessas tags com `TfidfVectorizer`;
3. A similaridade entre livros é calculada com **similaridade do cosseno** (`cosine_similarity`);
4. A partir dos 2 livros recomendados pela CF, o sistema procura livros **similares em conteúdo**, excluindo:
   - Livros já avaliados pelo usuário;
   - Livros já recomendados pela etapa colaborativa.

Dessa forma, são escolhidos **3 novos livros** com maior similaridade de conteúdo.

---

### 3. Recomendação Híbrida em Cascata

A lógica híbrida é implementada pela função:

```python
top_n_hibrida_cascata(user_id, n_cf=2, n_cb=3, model=knn)
