import pandas as pd
import polars as pl
import solution.preprocessing
import os
from solution.preprocessing.preprocessing import base_preprocessing

class ValidationPreparator:
    def transform(self, history_for_validation):

        # рассматриваем юзеров, у которых было хотя бы две сессии за период
        validation_users = history_for_validation.user_id.value_counts()
        validation_users = validation_users[validation_users >= 2].index

        val_entries = history_for_validation[history_for_validation.user_id.isin(validation_users)]

        # каждой сессии внутри пользователя присуждаем последовательный по хронологии номер
        sessioin_ranks = history_for_validation.sort_values('session_id').groupby('user_id').session_dt.rank().rename('val_session_rank')
        val_entries = val_entries.merge(sessioin_ranks, left_index=True, right_index=True)

        # подгонка под тест
        # в тесте минимальная длина сессии - 2 айтема, максимальная - 26
        # без фильтров 25й персентиль длин сессий получается 1 и много выбросов по 100-200-800 взаимодействий
        history_candidates = val_entries[(val_entries.vacancy_id.apply(len) > 1) & (val_entries.vacancy_id.apply(len) < 30)]

        # сессии для таргета: точно не первая сессия в тестовом периоде и в сессии есть отклик
        target_session_candidates = val_entries[(val_entries.val_session_rank > 1) & (val_entries.action_type.apply(lambda x: 1 in set(x)))]

        # достаем из второй сессии айтемы, действия и таймстампы
        target_session_candidates = target_session_candidates.explode('vacancy_id action_type action_dt'.split())
        # оставляем только отклики
        target_session_candidates = target_session_candidates[target_session_candidates.action_type == 1]
        # оставляем только первый по хронологии отклик
        target_candidates = target_session_candidates.sort_values('action_dt').groupby('user_id').head(1)
        # переименовываем найденые айтемы и даты в таргеты
        target_candidates = target_candidates.rename({'vacancy_id':'target', 'session_dt':'target_dt'}, axis=1)
        target_candidates = target_candidates['val_session_rank user_id target target_dt'.split()]

        # создаем датасет пар из предыдущей и последующей сессий
        target_candidates['val_session_rank'] = target_candidates['val_session_rank'] - 1
        candidate_session_pairs = history_candidates.merge(target_candidates, on='user_id val_session_rank'.split())
        # семплируем по одной сессии для каждого юзера полность случайно
        validation_sessions = candidate_session_pairs.groupby('user_id').sample(1)

        return validation_sessions

def read_parquet_to_pandas(path):
    return pl.read_parquet(path).to_pandas()

def get_dataset_storage_path(for_validation):
    base_data_path = 'data'
    if for_validation:
        path = os.path.join(base_data_path, 'validation')
    else:
        path = os.path.join(base_data_path, 'submit')
    return path

def get_train(for_validation=True):
    base_path = get_dataset_storage_path(for_validation)
    train_path = os.path.join(base_path, 'train.pq')
    return read_parquet_to_pandas(train_path)

def get_test(for_validation=True):
    base_path = get_dataset_storage_path(for_validation)
    test_path = os.path.join(base_path, 'test.pq')
    return read_parquet_to_pandas(test_path)

def store_solution(df, experiment_name, for_validation=True):
    
    base_path = 'data/submissions/'
    split_path = 'val' if for_validation else 'test'
    submission_path = os.path.join(base_path, split_path, f'{experiment_name}.pq')
    
    pl.from_pandas(df).write_parquet(submission_path)
    
    
    
if __name__ == '__main__':
    train_input = 'data/hh_recsys_train_hh.pq'
    test_input = 'data/hh_recsys_test_hh.pq'

    train_for_validation_output = 'data/validation/train.pq'
    validation_output = 'data/validation/test.pq'


    train = pl.read_parquet(train_input).to_pandas()
    test = pl.read_parquet(test_input).to_pandas()

    train_start_date = '2023-11-01'
    train_val_split_date = '2023-11-08'
    train_end_date = '2023-11-14'
    test_start_date = '2023-11-15'
    test_end_date = '2023-11-21'


    train = base_preprocessing.transform(train)
    test = base_preprocessing.transform(test)
    #train['session_dt'] = train['action_dt'].apply(lambda x: x[0])

    train_for_val = train[train['session_dt'] < train_val_split_date]
    val = train[train['session_dt'] >= train_val_split_date]

    val_prepare = ValidationPreparator()

    validation_dataset = val_prepare.transform(val)

    pl.from_pandas(train_for_val).write_parquet(train_for_validation_output)
    pl.from_pandas(validation_dataset).write_parquet(validation_output)

    train_for_submit_output = 'data/submit/train.pq'
    test_output = 'data/submit/test.pq'

    pl.from_pandas(train).write_parquet(train_for_submit_output)
    pl.from_pandas(test).write_parquet(test_output)