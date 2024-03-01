import pandas as pd
import polars as pl
import os
import logging
from solution.preprocessing.preprocessing import base_preprocessing
from solution.models.recommender_base import PipelineInfo, DatasetFabric, Splits

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

def read_parquet(path):
    return pl.read_parquet(path)

def get_dataset_storage_path(for_validation):
    base_data_path = 'data'
    if for_validation == Splits.VALIDATION:
        path = os.path.join(base_data_path, 'validation')
    elif for_validation == Splits.TEST:
        path = os.path.join(base_data_path, 'submit')
    else:
        path = os.path.join(base_data_path, 'micro')
    return path

def get_submission_storage_path(for_validation):
    if for_validation == Splits.VALIDATION:
        return 'val'
    elif for_validation == Splits.TEST:
        return 'test'
    else:
        return 'micro'

def get_train(for_validation=Splits.VALIDATION):
    base_path = get_dataset_storage_path(for_validation)
    train_path = os.path.join(base_path, 'train.pq')
    return read_parquet(train_path)

def get_test(for_validation=Splits.VALIDATION):
    base_path = get_dataset_storage_path(for_validation)
    test_path = os.path.join(base_path, 'test.pq')
    return read_parquet(test_path)

def get_vacancies(for_validation=Splits.VALIDATION):
    #base_path = get_dataset_storage_path(for_validation)
    base_path = './'
    vacancies_file_name = 'data/hh_recsys_vacancies_light.pq'
    return read_parquet(os.path.join(base_path, vacancies_file_name))
def store_solution(df, experiment_name, for_validation=Splits.VALIDATION):
    
    base_path = 'data/submissions/'
    split_path = get_submission_storage_path(for_validation)
    logging.info(f'{split_path}')
    submission_path = os.path.join(base_path, split_path, f'{experiment_name}.pq')
    
    df.write_parquet(submission_path)

def store_dataframe(df, model_name, for_validation=Splits.VALIDATION):
    base_path = 'data/dataframes/'
    split_path = get_submission_storage_path(for_validation)
    submission_path = os.path.join(base_path, split_path, f'{model_name}.pq')
    
    df.write_parquet(submission_path)

def load_dataframe(model_name, for_validation=Splits.VALIDATION):
    base_path = 'data/dataframes/'
    split_path = get_submission_storage_path(for_validation)
    submission_path = os.path.join(base_path, split_path, f'{model_name}.pq')
    
    return pl.read_parquet(submission_path)
    
class UserLogDataset(DatasetFabric):
    def get_train(self):
        return get_train(PipelineInfo.SPLIT)
    def get_test(self):
        return get_test(PipelineInfo.SPLIT)

class VacanciesDataset:
    def get(self):
        return get_vacancies(PipelineInfo.SPLIT)

def load_data(file_path):
    logging.info(f'Loading data from {file_path}')
    return pl.read_parquet(file_path).to_pandas()

def write_data(df, file_path):
    logging.info(f'Writing data to {file_path}')
    pl.from_pandas(df).write_parquet(file_path)  

def filter_data_by_date(df, start_date, end_date):
    return df[(df['session_dt'] >= start_date) & (df['session_dt'] < end_date)]

def transform_data(df):
    logging.info('Applying base preprocessing')
    return base_preprocessing.transform(df)

def prepare_dataset(train, train_start_date, train_end_date, 
                    val_start_date, val_end_date, output_train, output_val):
    train_for_val = filter_data_by_date(train, train_start_date, train_end_date)
    logging.info(f'Train for validation size: {train_for_val.shape}')
    val = filter_data_by_date(train, val_start_date, val_end_date)

    val_prepare = ValidationPreparator()
    validation_dataset = val_prepare.transform(val)
    logging.info(f'Validation dataset size: {validation_dataset.shape}')

    write_data(train_for_val, output_train)
    write_data(validation_dataset, output_val)

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    
    train_input = 'data/hh_recsys_train_hh.pq'
    test_input = 'data/hh_recsys_test_hh.pq'

    train_for_validation_output = 'data/validation/train.pq'
    validation_output = 'data/validation/test.pq'

    micro_train_for_validation_output = 'data/micro/train.pq'
    micro_validation_output = 'data/micro/test.pq'

    train_for_submit_output = 'data/submit/train.pq'
    test_for_submit_output = 'data/submit/test.pq'

    train_start_date = '2023-11-01'
    train_end_date = '2023-11-15'
    
    train_val_split_date = '2023-11-08'
    val_start_date = train_val_split_date
    val_end_date = train_end_date
    
    micro_train_val_split_date = '2023-11-03'
    micro_val_start_date = '2023-11-03'
    micro_val_end_date = '2023-11-05'

    test_start_date = '2023-11-15'
    test_end_date = '2023-11-21'

    train = load_data(train_input)
    test = load_data(test_input)

    train = transform_data(train)
    test = transform_data(test)
    
    logging.info('Validation dataset')
    #prepare_dataset(train, train_start_date, train_end_date, 
    #                val_start_date, val_end_date,
    #                train_for_validation_output, validation_output)

    logging.info('Micro validation dataset')
    prepare_dataset(train, train_start_date, micro_train_val_split_date,
                    micro_val_start_date, micro_val_end_date,
                    micro_train_for_validation_output, micro_validation_output)

    logging.info('Submit dataset')
    #write_data(train, train_for_submit_output)
    #write_data(test, test_for_submit_output)

