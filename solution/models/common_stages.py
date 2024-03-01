import polars as pl
import numpy as np
import pandas as pd
from solution.models.recommender_base import Reranker, UserHistoryIter,CandidateGenerator, HistoryFilter
import logging
class RandomRanker(Reranker):
    def __init__(self, name):
        self._name = name
    
    def score(self, candidates):
        return candidates.map_rows(lambda x: np.random.rand()).rename({'map':'score'})

class ConstantRanker(Reranker):
    def __init__(self, name):
        self._name = name
    
    def score(self, candidates):
        return candidates.with_columns(pl.lit(1).alias('score'))

class CatBoostLikeRanker(Reranker):
    def __init__(self, name, feature_list):
        self._name = name
        self._feature_list = feature_list
    
    def score(self, candidates):
        inference_dataset = candidates.select(*self._feature_list).to_pandas()
        inference_dataset['score'] = inference_dataset.apply(lambda x: np.random.rand(), axis=1)
        scored_candidates = pl.concat((candidates, pl.from_pandas(pd.DataFrame(inference_dataset['score']))),
            how='horizontal'
        )
        return scored_candidates

class HistoryIter(UserHistoryIter):
    def get_plain_history(self, user_histories):
        return user_histories.explode(
                'action_type',
                'vacancy_id',
                'action_dt'
            )
    def iter_queries(self, user_histories):
        for history_entry_id in range(len(user_histories)):
            entry = user_histories[history_entry_id]
            #logging.info(entry)
            user_id = entry['user_id'].item()
            yield user_id, self.get_plain_history(entry)

class ConstantCandidateGenerator(CandidateGenerator):
    def __init__(self, candidates):
        self._candidates = candidates
    def _get_candidates(self, user_history):
        return user_history.select('user_id').join(self._candidates, how='cross')


# Техдолг: этот класс можно переписать через обстрактный датафрейм из recommender_base
# а в специализации для polars оставить только конструктор
class I2IListModel(CandidateGenerator):
    def _prepare_anchor_column(self, user_history):
        return user_history.rename({'vacancy_id':'anchor_vacancy_id'})
    def _get_candidates(self, user_history):
        prepared_user_history = self._prepare_anchor_column(user_history)
        logging.info(prepared_user_history, self._i2i)
        candidates = prepared_user_history \
            .join(self._i2i, on='anchor_vacancy_id') \
            .explode('recom_vacancy_id', 'score') \
            .select(
                'user_id',
                pl.col('recom_vacancy_id').alias('vacancy_id'),
                'score'
            )
        aggregated_candidates = candidates.group_by('user_id', 'vacancy_id') \
            .agg(pl.max('score'))
        return aggregated_candidates

class CandGenWrapper(CandidateGenerator):
    def __init__(self, predictor):
        self._predictor = predictor
    def _get_candidates(self, df):
        return self._predictor.predict(df)

class HistoryFilterClicked(HistoryFilter):
    def __init__(self, user_history):
        self.fit(user_history)
    def fit(self, user_history):
        self._user_history = user_history.filter(pl.col('action_type') == 1)
        return self