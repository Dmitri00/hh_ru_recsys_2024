from abc import ABC, abstractmethod
from ast import Pass
from enum import Enum
import logging

from polars import DataFrame
import polars as pl
import pandas as pd
import numpy as np


candidates_schema = {"vacancy_id": pl.Int64, "score": pl.Int64}
user_history_schema = {
    'user_id': pl.Int64,
    'vacancy_id': pl.Int64,
    'action_type': pl.String,
    'action_dt': pl.String
}

# это базовая сущность, которая до сих пор зависит от сторонних фреймворков и
# эта сущность в процессе своего очищения:
# 1) всем остальным пайплайнам базового рекомендера нужно переписать табличные операторы с polars/pandas на данные методы
# 2) создать конкретную реализацию датафрейма на поларс
# 3) портировать конкретные стейджи пайплана на поларс на данный датафрейм
class UserDataset:
    def outer_join(self, other, on):
        pass
    def inner_join(self, other, on):
        pass
    def anti_join(self, other, on):
        pass
    def cross_join(self, other):
        pass
    def rename(self, col_mapping: dict):
        pass
    def sort(self, on, descending=True):
        pass
    def select(self, columns):
        pass
    def agg_lists_with_order(self, group_by, list_columns):
        pass
    def fill_null_column(self, column, fill_value):
        pass
    def group_by_and_agg(self, group_by, agg_func:str):
        pass
    def __length__(self):
        pass


class PipelineStage(ABC):
    @abstractmethod
    def predict(self, x: UserDataset) -> UserDataset:
        pass
    def set_name(self, name):
        self._name = name
        return self

class TrainableStage(PipelineStage):
    @abstractmethod
    def fit(self, train_df):
        pass

class HistoryFilter(TrainableStage):
    def predict(self, x):
        logging.info(f'Applying history filter')
        before_filter = len(x)
        this_user_history = self._user_history
        # портировать на абстрактный UserDataset.anti_join 
        filtered = x.join(this_user_history, on='user_id vacancy_id'.split(), how='anti')
        after_filter = len(filtered)
        logging.info(f'Before shown filter:{before_filter} rows, after filter:{after_filter} rows')
        return filtered
    @abstractmethod
    def fit(self, x: UserDataset):
        pass



class CandidateGenerator(PipelineStage):
    @abstractmethod
    def _get_candidates(self, user_history):
        pass
    def get_model_name(self):
        return self._name
    def predict(self, user_history):
        # эта функция для юзеров без рекомендаций должна возвращать строку с user_id, item_id=null
        candidates = self._get_candidates(user_history)
        # портировать на абстрактный UserDataset.group_by_and_agg(max) 
        candidates = candidates.group_by('vacancy_id user_id'.split(), maintain_order=True).max()
        # портировать на абстрактный UserDataset.rename 
        candidates = candidates.rename({'score': f'{self.get_model_name()}_score'})
        logging.info(f'Candidates from {self.get_model_name()}:{candidates}')
        return candidates


class HistoryFilterFromDataframe(HistoryFilter, TrainableStage):
    def __init__(self, user_history):
        self._user_history = user_history

class PipelineStageException(Exception):
    pass

class DatasetFabric(ABC):
    @abstractmethod
    def get_train(self, split: str):
        pass
    @abstractmethod
    def get_test(self, split: str):
        pass

class Splits(Enum):
    VALIDATION = 'val'
    TEST = 'test'

assert Splits.VALIDATION == Splits('val')

class UserHistoryIter(ABC):
    @abstractmethod
    def iter_queries(self, user_histories):
        pass



class Reranker(PipelineStage):
    def get_column_name(self):
        return self._name
    @abstractmethod
    def score(self, candidates):
        pass
    def predict(self, candidates):
        # портировать на абстрактный UserDataset.rename 
        scored_candidates = self.score(candidates).rename({'score': self.get_column_name()})
        
        # портировать на абстрактный UserDataset.sort 
        ranked_candidates = scored_candidates.sort(self.get_column_name(), descending=True)
        logging.info(ranked_candidates)
        return ranked_candidates



class RecomPack(PipelineStage):
    def __init__(self, name):
        self._name = name
    def predict(self, user_recoms):
        logging.info(f'Packing recoms {user_recoms}')
        # портировать на абстрактный UserDataset.agg_lists_with_order 
        recoms = user_recoms.group_by('user_id', maintain_order=True).agg(pl.col('vacancy_id'))
        return recoms

class SubmitPrepare(PipelineStage):
    def __init__(self, submit_dataframe):
        self._submit_dataframe = submit_dataframe.select('user_id', 'session_id')

    def fill_null_list(self, col):
        return pl.when(col.is_not_null()).then(col).otherwise([])#.to_series()
    def predict(self, recoms):
        # когда предыдущие стейджи перестанут забывать юзеров без рекомов и
        # будут обозначать их значением null в vacancy_id, то
        # о прокидывании теста в данный класс можно будет забыть и класс очистится
        # портировать на абстрактный UserDataset.select и UserDataset.outer_join
        submit = self._submit_dataframe.join(recoms.select('user_id', pl.col('vacancy_id').alias('predictions')), on='user_id', how='outer_coalesce')
        #logging.info(submit)

        # портировать на абстрактный UserDataset.fill_null
        submit = submit.with_columns(self.fill_null_list(pl.col('predictions')))        

        return submit


class Pipeline(ABC):
    @abstractmethod
    def reset_stages(self):
        pass
    def add_stage(self, stage: PipelineStage):
        self._stages.append(stage)
    def predict(self, x: UserDataset):
        for stage in self._stages:
            try:
                x = stage.predict(x)
                logging.info(x)
            except PipelineStageException as e:
                logging.exception(e)
                raise PipelineStageException() from e
        return x
    
    def fit(self, x: UserDataset):
        for stage in self._stages:
            try:
                if isinstance(stage, TrainableStage) or isinstance(stage, Pipeline):
                    logging.info(f'Training stage {stage}')
                    stage.fit(x)
            except PipelineStageException as e:
                logging.exception(e)
                raise PipelineStageException() from e
        return self



class ListPipeline(Pipeline):
    def __init__(self):
        self.reset_stages()
    def reset_stages(self):
        self._stages = []


class JoinCandidatesStage(ListPipeline):
    def __init__(self, parallel_stages, join_by):
        self._join_by = join_by
        self._stages = parallel_stages
    def predict(self, x):
        results = None
        for i, stage in enumerate(self._stages):
            #logging.info(results)
            this_stage_output = stage.predict(x)
            if results is not None:
                # портировать на абстрактный UserDataset.outer_join
                results = results.join(this_stage_output, on=self._join_by, how='outer_coalesce')
            else:
                results = this_stage_output
            logging.info(f'joint candidates # {i}: {results}')
        return results