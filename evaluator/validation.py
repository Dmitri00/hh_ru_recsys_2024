import pandas as pd
import polars as pl

def prepare_validation_first_two_sessions(history_for_validation):
    # оставить юзеров, у которых был последним отклик
    # во второй сессии был отклик

    # юзера, у которых было хотя бы две сессии за период
    validation_users = history_for_validation.user_id.value_counts()
    validation_users = validation_users[validation_users >= 2].index

    val_entries = history_for_validation[history_for_validation.user_id.isin(validation_users)]

    # оставить только первые две сессии в валидационном периоде
    sessioin_ranks = val_entries.sort_values('session_id').groupby('user_id').session_dt.rank().rename('val_session_rank')
    sessioin_ranks = sessioin_ranks[sessioin_ranks<= 2]

    val_entries = val_entries.merge(sessioin_ranks, left_index=True, right_index=True)


    # каждую первую сессию просто откладываем, это будет Х
    story_session = val_entries[val_entries.val_session_rank == 1]

    # каждую вторую сессию будем препроцессить, это будет Y
    target_session = val_entries[val_entries.val_session_rank == 2]

    # достаем из второй сессии айтемы, действия и таймстампы
    target_interactions = target_session.explode('vacancy_id action_type action_dt'.split())
    # оставляем только отклик отклики
    target_interactions = target_interactions[target_interactions.action_type == 1]
    # оставляем только первый по хронологии отклик
    target_otclick = target_interactions.sort_values('action_dt').groupby('user_id').head(1)
    # переименовываем найденые айтемы и даты в таргеты
    target_otclick = target_otclick['vacancy_id action_dt user_id'.split()] \
        .rename({'vacancy_id':'target', 'action_dt': 'target_dt'}, axis=1)

    # джоним сессии пользовательской истории (X) с таргетом (Y)
    validation_dataset = story_session.merge(target_otclick, on='user_id')

    # подгонка под тест
    # в тесте минимальная длина сессии - 2 айтема, максимальная - 25
    # без фильтров 25й персентиль длин сессий получается 1 и много выбросов по 100-200-800 взаимодействий
    validation_dataset = validation_dataset[
        (validation_dataset.vacancy_id.apply(len)>1)
    &
       (validation_dataset.vacancy_id.apply(len)< 30) ]
    return validation_dataset
    

train_input = 'data/hh_recsys_train_hh.pq'

train_for_validation_output = 'data/validation/train.pq'
validation_output = 'data/validation/test.pq'

train = pl.read_parquet(path).to_pandas()

train_start_date = '2023-11-01'
train_val_split_date = '2023-11-08'
train_end_date = '2023-11-14'
test_start_date = '2023-11-15'
test_end_date = '2023-11-21'


train['session_dt'] = train['action_dt'].apply(lambda x: x[0])

train_for_val = train[train['session_dt'] < train_val_split_date]
val = train[train['session_dt'] >= train_val_split_date]

validation_dataset = prepare_validation_first_two_sessions(val)

pl.from_pandas(train_for_val).write_parquet(train_for_validation_output)
pl.from_pandas(validation_dataset).write_parquet(validation_output)

