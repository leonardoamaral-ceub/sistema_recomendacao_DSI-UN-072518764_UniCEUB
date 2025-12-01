# main.py
import os
import pandas as pd
import numpy as np
import dill

from fastapi import FastAPI
from pydantic import BaseModel

from surprise import Reader, Dataset
from surprise.prediction_algorithms.knns import KNNBaseline
from surprise.model_selection import train_test_split
from surprise import accuracy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "notebooks")

# MODELOS DE ENTRADA (REQUEST BODY)

class UserRequest(BaseModel):
    user_id: int
    n: int | None = 5  # quantidade padrão de recomendações


class TopNRequest(BaseModel):
    n: int = 10


class HibridaRequest(BaseModel):
    user_id: int
    n_cf: int = 2
    n_cb: int = 3

# INICIALIZAÇÃO DO FASTAPI
app = FastAPI(
    title="Recommender Engine - Livros",
    description="API do Sistema de Recomendação (Filtragem Colaborativa, Conteúdo e Híbrido em Cascata)",
    version="1.0.0"
)

recomenda_obj = None
livros_df = None
ratings_df = None
tags_df = None
book_tags_df = None

knn = None
trainset = None
testset = None

def load_datasets_and_model():
    global recomenda_obj, livros_df, ratings_df, tags_df, book_tags_df
    global knn, trainset, testset, livros_ratings
    
    def load_csv(filename):
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            try:
                return pd.read_csv(path)
            except Exception as e:
                return None
        else:
            return None
    
    livros_df = load_csv("books.csv")
    ratings_df = load_csv("ratings.csv")
    tags_df = load_csv("tags.csv")
    book_tags_df = load_csv("book_tags.csv")
        
    livros_ratings = ratings_df.merge(livros_df, on="book_id", how="left")
    
    try:
        reader = Reader(rating_scale=(0, 5))
        data = Dataset.load_from_df(
            ratings_df[["user_id", "book_id", "rating"]],
            reader
        )

        trainset, testset = train_test_split(data, test_size=0.25, random_state=42)
        
        sim_options = {"name": "pearson_baseline", "user_based": False}

        knn = KNNBaseline(k=33, sim_options=sim_options)
        knn.fit(trainset)
    except Exception as e:
        knn = None
        trainset = None
        testset = None

# Executa inicialização ao iniciar FastAPI
load_datasets_and_model()

def recomenda_livro_cf(user_id, book_id):
    global livros_df, livros_ratings, knn, trainset

    # ID do usuário para predição
    uid = user_id
    # ID do filme para predição
    iid = book_id
    nome_livro = livros_df.query('book_id == @book_id')['original_title'].values[0]
    avaliacao = None
    if livros_ratings.query('user_id == @user_id and book_id == @book_id')['original_title'].values.size == 0:
        avaliacao = 'Usuário não avaliou o livro!'
    else:
        nota_livro = ratings_df.query('user_id == @user_id and book_id == @book_id')['rating'].values[0]
        avaliacao = 'Avaliação do usuário:', nota_livro

    return {
        'Livro:': nome_livro,
        'Usuário:': user_id,
        'Avaliacao': avaliacao,
        'Estimativa de Avaliação[0-5]:': round(knn.predict(trainset.to_raw_uid(uid), trainset.to_raw_iid(iid))[3], 2)
    }

if livros_df is not None and ratings_df is not None:
    livros_ratings = ratings_df.merge(livros_df, on='book_id', how='left')
    
def top_n(user_id,n):
  # Selecionando apenas os livros do treinamento
  lista_livros_treino = []
  for x in trainset.all_items():
    lista_livros_treino.append(trainset.to_raw_iid(x))
  # Selecionando os livros do treinamento que o usuário não avaliou
  livros_ratings_user = ratings_df.query('user_id == @user_id')['book_id'].values
  livros_ratings_user_nao = livros_df.query('book_id not in @livros_ratings_user')
  livros_ratings_user_nao = livros_ratings_user_nao.query('book_id in @lista_livros_treino')['book_id'].values
  # Criando um ranking para o usuário para os livros não avaliados
  ranking=[]
  for book_id in livros_ratings_user_nao:
    ranking.append((book_id, knn.predict(trainset.to_inner_uid(user_id), trainset.to_inner_iid(book_id))[3]))
  # Ordenando os TOP livros avaliados
  ranking.sort(key=lambda x: x[1], reverse=True)
  # Selecionando os Ids dos livros
  x,_ = zip(*ranking[:n])
  # Listando os nomes dos livros em ordem de recomendação
  return livros_df.query('book_id in @x')['original_title'].copy().reset_index(drop=True)

class Recomenda:
    def __init__(self, ratings, livros, trainset):
        self.ratings = ratings
        self.movies = livros
        self.trainset = trainset

    def top_n(self,user_id,n):
      # Selecionando apenas os livros do treinamento
      lista_livros_treino = []
      for x in trainset.all_items():
        lista_livros_treino.append(trainset.to_raw_iid(x))
      # Selecionando os livros do treinamento que o usuário não avaliou
      livros_ratings_user = ratings_df.query('user_id == @user_id')['book_id'].values
      livros_ratings_user_nao = livros_df.query('book_id not in @livros_ratings_user')
      livros_ratings_user_nao = livros_ratings_user_nao.query('book_id in @lista_livros_treino')['book_id'].values
      # Criando um ranking para o usuário para os livros não avaliados
      ranking=[]
      for book_id in livros_ratings_user_nao:
        ranking.append((book_id, knn.predict(trainset.to_inner_uid(user_id), trainset.to_inner_iid(book_id))[3]))
      # Ordenando os TOP livros avaliados
      ranking.sort(key=lambda x: x[1], reverse=True)
      # Selecionando os Ids dos livros
      x,_ = zip(*ranking[:n])
      # Listando os nomes dos livros em ordem de recomendação
      return livros_df.query('book_id in @x')['original_title'].copy().reset_index(drop=True)
  
class Recomenda:
    def __init__(self, livros_ratings):
        self.livros_ratings = livros_ratings

    def top_n(self,n):
      # Quantidade de Avaliações TOP5 Usuários
      return {
          'Top_N': self.livros_ratings['user_id'].value_counts().head(n).to_dict()
      }
      
# FUNÇÃO TOP N HÍBRIDA POR CASCATA (SEQUENCIAL) 

# Juntando 'book_tags' com 'tags' para obter os nomes das tags
book_tags_nomes = book_tags_df.merge(tags_df, on='tag_id')

# Agrupando as tags por livro (goodreads_book_id) em uma única string
book_tag_counts = book_tags_nomes.groupby('goodreads_book_id')['tag_name'].apply(lambda x: ' '.join(x)).reset_index()
book_tag_counts.columns = ['book_id', 'all_tags']

# Unindo com o DataFrame principal de livros
livros_para_cb = livros_df.merge(book_tag_counts, on='book_id')
livros_para_cb = livros_para_cb[['book_id', 'original_title', 'all_tags']].copy()
livros_para_cb = livros_para_cb.fillna('')

# Criando a matriz TF-IDF com base nas tags de todos os livros
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(livros_para_cb['all_tags'])
print('Tamanho da Matriz TF-IDF (Livros x Tags):', tfidf_matrix.shape)

# Calculando a similaridade do cosseno
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print('Tamanho da Matriz de Similaridade de Cosseno:', cosine_sim.shape)

# Mapeamentos para buscar dados rapidamente
bookid_to_index = pd.Series(livros_para_cb.index, index=livros_para_cb['book_id']).drop_duplicates()
index_to_bookid = pd.Series(livros_para_cb['book_id'], index=livros_para_cb.index).drop_duplicates()

print("Mapeamentos criados.")

def top_n_hibrida_cascata(user_id, n_cf=2, n_cb=3, model=knn):
    # (Códigos das linhas 1 a 18 da função original)

    # 1.1. Encontrar livros não avaliados 
    rated_books_ids = set(ratings_df.query('user_id == @user_id')['book_id'].values)
    all_books_ids = set(livros_df['book_id'].values)
    unrated_books_ids = list(all_books_ids - rated_books_ids)
    
    # 1.2. Criar dataset de teste apenas com IDs que o modelo conhece 
    unrated_books_test_set = []
    
    for book_id in unrated_books_ids:
        # A. Verifica se o livro existe na base de Conteúdo (CB)
        if book_id not in bookid_to_index.index:
            continue
            
        # B. Verifica se o livro (raw ID) é conhecido pelo modelo KNN (CF)
        # O método .knows_item() requer o ID interno, o que é problemático.
        # A maneira mais direta e correta é verificar se o raw ID está no dicionário de mapeamento
        # interno-para-bruto (ou vice-versa), que é mais robusto:
        if book_id in model.trainset._raw2inner_id_items:
             unrated_books_test_set.append((user_id, book_id, 4)) # 4 é rating dummy
             
    # 1.3. Fazer predições e selecionar o Top n_cf
    predictions = model.test(unrated_books_test_set)
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_cf_predictions = predictions[:n_cf]
    
    cf_book_ids = [pred.iid for pred in top_cf_predictions]
    cf_titles = livros_df.query('book_id in @cf_book_ids')['original_title'].values.tolist()
    
    # 2. ETAPA CB: Recomendação Baseada em Conteúdo (Top n_cb)
      
    # 2.1. Lista de IDs a serem excluídos do resultado final do CB
    excluded_ids = set(cf_book_ids) | rated_books_ids
    
    # 2.2. Encontrar os índices na matriz de similaridade para os n_cf livros (Seeds)
    seed_indices = [bookid_to_index[b_id] for b_id in cf_book_ids if b_id in bookid_to_index.index]
    
    if not seed_indices:
        print("Aviso: Os livros recomendados por CF não foram encontrados na base de conteúdo. Retornando apenas CF.")
        return pd.Series(cf_titles)

    cb_scores = {}
    
    # 2.3. Iterar sobre todos os livros (candidatos) para calcular o score CB
    for i in livros_para_cb.index:
        candidate_book_id = index_to_bookid[i]
        
        # Ignorar livros já recomendados pelo CF ou já avaliados
        if candidate_book_id in excluded_ids:
            continue
            
        # Calcula a similaridade máxima com os livros "seed" (o max é uma heurística comum para o CB)
        max_similarity = np.max(cosine_sim[i, seed_indices])
        
        # Armazena o score CB (similaridade)
        cb_scores[candidate_book_id] = max_similarity

    # 2.4. Selecionar o Top n_cb
    sorted_cb_scores = sorted(cb_scores.items(), key=lambda item: item[1], reverse=True)
    top_cb_book_ids = [item[0] for item in sorted_cb_scores[:n_cb]]
    
    cb_titles = livros_df.query('book_id in @top_cb_book_ids')['original_title'].values.tolist()
    
    # 3. RESULTADO FINAL (UNIÃO)
    final_titles = cf_titles + cb_titles
    
    return pd.Series(final_titles)

# ENDPOINT 1 — dataset_health
@app.get("/dataset_health", tags=["dataset"])
def dataset_health():
    return {
        "livros": {
            "status": "OK - " + str(len(livros_df)) + " linhas" if livros_df is not None else "NOT LOADED",
        },
        "ratings": {
            "status": "OK - " + str(len(ratings_df)) + " linhas" if ratings_df is not None else "NOT LOADED",
        },
        "tags": {
            "status": "OK - " + str(len(tags_df)) + " linhas" if tags_df is not None else "NOT LOADED",
        },
        "book_tags": {
            "status": "OK - " + str(len(book_tags_df)) + " linhas" if book_tags_df is not None else "NOT LOADED",
        }
    }

# ENDPOINT 2 — recomenda_livro
@app.post("/recomenda_livro", tags=["recomendacao"])
def recomenda_livro(req: UserRequest):
    return recomenda_livro_cf(req.user_id, 5)

# ENDPOINT 3 — top_n
@app.post("/top_n", tags=["analise"])
def top_n(req: TopNRequest):
    recomenda = Recomenda(livros_ratings).top_n(req.n)
    return recomenda

# ENDPOINT 4 — top_n_hibrida_cascata
@app.post("/top_n_hibrida_cascata", tags=["hibrida"])
def top_n_hibrida(req: HibridaRequest):
    recs_cascata = top_n_hibrida_cascata(user_id=req.user_id, n_cf=req.n_cf, n_cb=req.n_cb)
    return recs_cascata