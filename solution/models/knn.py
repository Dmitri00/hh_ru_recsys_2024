
import polars as pl
import logging
import numpy as np
from solution.models.model_base import TrainPredictModelApp
import sys
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline as sk_Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from itertools import zip_longest
from sklearn.feature_extraction.text import CountVectorizer
import faiss
from sklearn.preprocessing import LabelEncoder

from solution.preprocessing.validation import VacanciesDataset, UserLogDataset, store_dataframe
from solution.models.common_stages import CandGenWrapper
from solution.models.recommender_base import Splits

class DictLabelEncoder:
    def fit(self, labels):
        idxs = list(range(len(labels)))
        labels = labels.to_list()
        self._item2idx = dict(zip(labels, idxs))
        self._idx2item = dict(zip(idxs, labels))
        self.classes_ = labels
    def transform(self, labels):
        result = list()
        for label in labels:
            if label in self._item2idx:
                result.append(self._item2idx[label])
            else:
                result.append(None)
                logging.info(f'Неизвестная категория {label}')
        return result
    def inverse_transform(self, idxs):
        return [self._idx2item[idx] for idx in idxs if idx  >= 0 and idx < len(self._idx2item)]

class ItemScoreAggregator:
    def transform(self, items, scores):
        return self._aggregate(items, scores)


class UniqueItemAggregator(ItemScoreAggregator):
    def _aggregate(self, items, scores):
        items = items.flatten()
        scores = scores.flatten()
        return list(set(items))
class RoundRobinAggregator(ItemScoreAggregator):
    def _aggregate(self, items, scores):
        seen_items = set([-1])
        recoms = list()
        for items_iter in zip_longest(*items):
            for item in items_iter:
                if item not in seen_items and item is not None:
                    recoms.append(item)
                    seen_items.add(item)
                else:
                    continue
        return recoms
            
class PCACountVectorizer:
    def __init__(self, n_components=7):
        self._pipeline = sk_Pipeline([('counter',CountVectorizer(input='content', tokenizer=lambda x: list(x), lowercase=False)),
                                     ('pca', TruncatedSVD(n_components=n_components))])
    def fit(self, items_tokens):
        #item_embeds = self._counter.fit(items_tokens)
        return self._pipeline.fit(items_tokens)
    def transform(self, item_tokens):
        return self._pipeline.transform(item_tokens)

from implicit.als import AlternatingLeastSquares

class ALSItem2Vec:
    def __init__(self, factors=20, regularization=0.1, use_native=True, use_cg=True):
        self.factors = factors
        self.regularization = regularization
        self.use_native = use_native
        self.use_cg = use_cg
        self.model = None
        
    def fit(self, X, *args, **kwargs):
        self.model = AlternatingLeastSquares(factors=self.factors, 
                                             regularization=self.regularization,
                                             use_native=self.use_native,
                                             use_cg=self.use_cg,
                                             iterations=10)
        self.model.fit(X)
        return self
        
    def transform(self, item_tokens):
        return self.model.user_factors
        
class ALSCountVectorizer:
    def __init__(self, n_components=7):
        self._pipeline = sk_Pipeline([('counter',CountVectorizer(input='content', tokenizer=lambda x: list(x), lowercase=False)),
                                     ('als', ALSItem2Vec(factors=n_components))])
    def fit(self, items_tokens):
        return self._pipeline.fit(items_tokens)
    def transform(self, item_tokens):
        return self._pipeline[1][0].model.user_factors

class ClusterKnn:
    def __init__(self, top_size=300, n_clusters=100):
        self._knn = NearestNeighbors()
        self._kmeans = KMeans(n_clusters)
        self._topk = top_size

    def fit(self, item_embeds):
        
        self._knn.fit(item_embeds)
        
        self._kmeans.fit(item_embeds)
        
        centroids = self._kmeans.cluster_centers_
        
        items, scores = self._knn.kneighbors(centroids, self._topk)
        
        self._items = items
        self._scores = scores
        
        
    def kneighbors(self, item_embeds, k):
        centroid_idxs = self._kmeans.predict(item_embeds)
        
        items = []
        scores = []
        
        for centroid_idx in centroid_idxs:
            items.append(self._items[centroid_idx])
            scores.append(self._scores[centroid_idx])
        return np.array(items), np.array(scores)
class FaissKnn:
    def __init__(self, ndims):
        self._index = faiss.IndexFlatL2(ndims)

    def fit(self, item_embeds):
        
        self._knn.add(item_embeds)
        return self
        
    def kneighbors(self, item_embeds, k):
        scores, items = self._index.search(item_embeds, k)
        
        return scores, items
    
class FastFaissKnn(FaissKnn):
    def __init__(self, ndims, nlist, nprobe):
        quantizer = faiss.IndexFlatL2(ndims)  # the other index
        self._index = faiss.IndexHNSWFlat(ndims, 32, faiss.METRIC_INNER_PRODUCT)
        self._index.nprobe = nprobe
    
    def fit(self, item_embeds):
        #item_embeds = item_embeds.astype('float64')
        self._index.train(item_embeds)
        self._index.add(item_embeds)  
        
        return self

class KnnModel:
    def __init__(self, knn_model, item_embedder, item_aggregator=RoundRobinAggregator()):
        self._knn = knn_model
        self._embedder = item_embedder
        self._key_encoder = DictLabelEncoder()
        self._item_aggregator = item_aggregator
    def fit(self, user_item_interactions):
        user_item_interactions = user_item_interactions.sort('key')
        self._key_encoder.fit(user_item_interactions['key'])
        item_tokens = user_item_interactions['tokens']
        logging.info('fitting embedder')
        item_vectors = self._embedder.fit(item_tokens).transform(item_tokens)
        logging.info('fitting knn index')
        normed_vectors = item_vectors/np.linalg.norm(item_vectors, axis=1).reshape(-1,1)
        logging.info(f'normed vectros shape {normed_vectors.shape}')
        self._uv_mat = normed_vectors
        self._knn.fit(self._uv_mat)
        self._key_encoder_classes = set(self._key_encoder.classes_)
        return self
    def predict(self, users_items, kneib=20):
        existing_items = list(set(users_items).intersection(self._key_encoder_classes))
        if existing_items == []:
            return []
        item_indexes = self._key_encoder.transform(existing_items)
        item_embeds = self._uv_mat[item_indexes]
        #item_embeds = self._embedder.transform()
        scores, items = self._knn.kneighbors(item_embeds, kneib)
        items = self._item_aggregator.transform(items, scores)
        items = self._key_encoder.inverse_transform(items)
        
        return items

class I2IListKnnModel(KnnModel):
    def fit(self, user_item_interactions):
        user_item_interactions = user_item_interactions.sort('key')
        logging.info('fitting knn model itself')
        super().fit(user_item_interactions)
        self._i2i_list = []
        
        logging.info('fitting i2i list')
        for item_id, item_embed in tqdm(zip(user_item_interactions['key'], self._uv_mat), total=len(user_item_interactions)):
            scores, items = self._knn.kneighbors(item_embed.reshape(1,-1), 100)
            query_item_index = self._key_encoder.transform([item_id])[0]
            #import pdb; pdb.set_trace()

            if len(items) == 0:
                continue
            
            scores = scores[0]
            items = items[0]
            assert len(scores) == len(items), f'{len(scores)} len(scores)!= len(items) {len(items)}'
            
            found_mask = items != -1
            scores = scores[found_mask]
            items = items[found_mask]
            
            dublicate_mask = items != query_item_index
            scores = scores[dublicate_mask]
            items = items[dublicate_mask]
            recom_item_remaped = self._key_encoder.inverse_transform(items)
            scores = scores.tolist()
            if len(recom_item_remaped) == 0:
                continue
            assert len(items) == len(recom_item_remaped), f'{items} len(scores)!= len(items) {recom_item_remaped}'
            self._i2i_list.append(
                {'anchor_vacancy_id': item_id, 'recom_vacancy_id': recom_item_remaped,
                    'score': scores })
                
        self._i2i_list = pl.DataFrame(self._i2i_list)
        logging.info(f'i2i list {self._i2i_list}')
        return self
    
    def _single_user_query(self, user_items, kneib=20):
        user_items = user_items.rename({'vacancy_id':'anchor_vacancy_id'})
        scored_candidates = user_items.join(self._i2i_list, on='anchor_vacancy_id') \
            .explode('recom_vacancy_id', 'score') \
            .select('user_id', 'session_id', pl.col('recom_vacancy_id').alias('vacancy_id'), 'score') \
            .group_by('user_id', 'vacancy_id').agg(pl.max('score'))
        topk_scored = scored_candidates \
            .filter(
                pl.col('score').rank(method='dense', descending=True).over('user_id') <=
                kneib)
            
        return topk_scored
        
    def predict(self, user_items, kneib=20):
        return self._single_user_query(user_items, kneib)

    def get_i2i_dataframe(self):
        return self._i2i_list
    def set_i2i_dataframe(self, i2i):
        self._i2i_list = i2i

class ItemGeoKnnModelWithDatasetPrepare:
    def transform(self, train, vacancies_dataframe):
        positives = train \
                .join(
                vacancies_dataframe.select('vacancy_id', 'area.id', 'area.regionId'), on='vacancy_id'
            ) \
                .with_columns(geo_user=pl.concat_str([pl.col('user_id'), pl.col('area.id'), pl.col('area.regionId')]))
        
        min_item_count = 1
        positives = self.hot_item_filter(positives, 'user_id', min_item_count)

        positives = positives.with_columns(user_id_remap = LabelEncoder().fit_transform(positives['user_id']))
        positives = positives.with_columns(user_id_geo_remap = LabelEncoder().fit_transform(positives['geo_user']))
        positives = positives.with_columns(vacancy_id_remap = LabelEncoder().fit_transform(positives['vacancy_id']))
        
        item_user_geo_interactions = positives.select('vacancy_id', 'user_id_remap') \
            .group_by('vacancy_id').agg(pl.col('user_id_remap').alias('user_id'))

        item_knn_dataset = item_user_geo_interactions.select(pl.col('vacancy_id').alias('key'), pl.col('user_id').alias('tokens'))
        return item_knn_dataset

    def hot_item_filter(self, positives, by, min_item_count):
        hot_vacancies = positives.group_by('vacancy_id').agg(pl.count(by).alias('hot_vacancy_count'))
        logging.info(f"Number of rows before count filter: {len(hot_vacancies)}")
        hot_vacancies = hot_vacancies.filter(hot_vacancies['hot_vacancy_count'] >= min_item_count) \
                    .select('vacancy_id')
        logging.info(f"Number of rows after count filter: {len(hot_vacancies)}")

        positives = positives.join(hot_vacancies, on='vacancy_id')
        return positives

        


class OnTheFlyKnnModel:
    def __init__(self, knn_model, item_embedder, item_storage):
        self._knn = knn_model
        self._embedder = item_embedder
        self._key_encoder = LabelEncoder()
        self._item_aggregator = UniqueItemAggregator()
        self._item_storage = item_storage
    def fit(self, user_item_interactions):
        user_item_interactions = user_item_interactions.sort('key')
        self._key_encoder.fit(user_item_interactions['key'])
        item_tokens = user_item_interactions['tokens']
        item_vectors = self._embedder.fit(item_tokens).transform(item_tokens)
        self._knn.fit(item_vectors)
        return self
    def predict(self, users_items, kneib=20):
        item_tokens = self._item_storage.search(users_items)['name']
        item_embeds = self._embedder.transform(item_tokens)
        scores, items = self._knn.kneighbors(item_embeds, kneib)
        items = self._item_aggregator.transform(items, scores)
        items = self._key_encoder.inverse_transform(items)        
        return items
        
class ItemTextStorage:
    def __init__(self, storage):
        self._storage = storage
    def search(self, items):
        mask = self._storage['vacancy_id'].is_in(items)        
        return self._storage.filter(mask)
    
def get_basic_knn():
    return KnnModel(
        knn_model=NearestNeighbors(),
        item_embedder=CountVectorizer(input='content', tokenizer=lambda x: list(x), lowercase=False),
        item_aggregator=RoundRobinAggregator()
    )

def get_i2i_list_knn():
    return I2IListKnnModel(
        knn_model=NearestNeighbors(),
        item_embedder=CountVectorizer(input='content', tokenizer=lambda x: list(x), lowercase=False),
        item_aggregator=RoundRobinAggregator()
    )

def get_pca_knn():
    return KnnModel(
        knn_model= NearestNeighbors(),
        item_embedder=PCACountVectorizer(n_components=64)
    )

def get_pca_faiss_knn():
    n_components = 64
    return KnnModel(
        knn_model= FaissKnn(n_components),
        item_embedder=PCACountVectorizer(n_components=n_components)
    )

def get_pca_fast_faiss_knn():
    n_components = 64
    nlists = 100
    nprobe = 1
    return KnnModel(
        knn_model= FastFaissKnn(n_components, nlists, nprobe),
        item_embedder=PCACountVectorizer(n_components=n_components)
    )
def get_i2i_pca_fast_faiss_knn():
    n_components = 64
    nlists = 100
    nprobe = 1
    return I2IListKnnModel(
        knn_model=FastFaissKnn(n_components, nlists, nprobe),
        item_embedder=PCACountVectorizer(n_components=64),
        item_aggregator=RoundRobinAggregator()
    )

def get_item_geo_i2i_pca_fast_faiss_knn(for_validation=True):
    n_components = 64
    nlists = 100
    nprobe = 1
    return  ItemGeoKnnModelWithDatasetPrepare(get_vacancies(for_validation),
        I2IListKnnModel(
            knn_model=FastFaissKnn(n_components, nlists, nprobe),
            item_embedder=PCACountVectorizer(n_components=64),
            item_aggregator=RoundRobinAggregator()
        )
    )

def get_name_knn():
    return OnTheFlyKnnModel(
        knn_model=ClusterKnn(top_size=50, n_clusters=2000),
        item_embedder=CountVectorizer(input='content'),
        item_storage=ItemTextStorage(item_descr)
    )

def get_name_knn_2():
    return KnnModel(
        knn_model=NearestNeighbors(),
        item_embedder=CountVectorizer(input='content')
    )

class ItemKnnModel(TrainPredictModelApp):
    def get_args(self):
        try:
            split = sys.argv[1]
            for_validation = TrainPredictModelApp.parse_split(split)
            exp_name = sys.argv[2]
            self._exp_name = exp_name
        except:
            for_validation = Splits('val')
        return {'for_validation': for_validation,
               'experiment_name':self._exp_name}
    def get_model(self, args):
        vacancies_dataset = VacanciesDataset().get()
        content_data_prepare = self._dataset_prepare
        train = UserLogDataset().get_train() \
            .explode('action_type', 'vacancy_id', 'action_dt')
        geo_knn_dataset = content_data_prepare.transform(train, vacancies_dataset)
        return CandGenWrapper(self._model.fit(geo_knn_dataset)).set_name(self._exp_name)
    def after_submit(self, args):
        
        i2i_dataframe = self._model.get_i2i_dataframe()
        store_dataframe(i2i_dataframe, self._exp_name, args['for_validation'])

# Подвязка к пайплайну рекомендаций
class ItemGeoKnnFaissApp(ItemKnnModel):
    def __init__(self):
        n_components = 64
        nlists = 100
        nprobe = 1
        self._model = I2IListKnnModel(
                knn_model=FastFaissKnn(n_components, nlists, nprobe),
                item_embedder=PCACountVectorizer(n_components=64),
                item_aggregator=RoundRobinAggregator()
            )
        self._dataset_prepare = ItemGeoKnnModelWithDatasetPrepare()

class ItemGeoKnnFaissFullApp(ItemKnnModel):
    def __init__(self):
        n_components = 32
        nlists = 100
        nprobe = 1
        self._model = I2IListKnnModel(
                knn_model=FastFaissKnn(n_components, nlists, nprobe),
                item_embedder=PCACountVectorizer(n_components=n_components),
                item_aggregator=RoundRobinAggregator()
            )
        self._dataset_prepare = ItemGeoKnnModelWithDatasetPrepare()

class ALSFaissFullApp(ItemKnnModel):
    def __init__(self):
        n_components = 16
        nlists = 100
        nprobe = 1
        self._model = I2IListKnnModel(
                knn_model=FastFaissKnn(n_components, nlists, nprobe),
                item_embedder=ALSCountVectorizer(n_components=n_components),
                item_aggregator=RoundRobinAggregator()
            )
        self._dataset_prepare = ItemGeoKnnModelWithDatasetPrepare()

class ALS32FaissFullApp(ItemKnnModel):
    def __init__(self):
        n_components = 32
        nlists = 100
        nprobe = 1
        self._model = I2IListKnnModel(
                knn_model=FastFaissKnn(n_components, nlists, nprobe),
                item_embedder=ALSCountVectorizer(n_components=n_components),
                item_aggregator=RoundRobinAggregator()
            )
        self._dataset_prepare = ItemGeoKnnModelWithDatasetPrepare()

        
if __name__ == '__main__':
    
    ALS32FaissFullApp().main()