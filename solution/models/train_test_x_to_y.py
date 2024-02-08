sessions_from_val = validation_sessions['user_id session_id vacancy_id'.split()]
submission_draft = als_train_for_val['user_id vacancy_id'.split()].rename({'vacancy_id':'predictions'}, axis=1) \
    .explode('predictions') \
    .groupby('user_id') \
    .predictions.agg(lambda x: list(set(x))) \
    .reset_index() \
    .merge(sessions_from_val, on='user_id')


submission_draft = als_train_for_val['user_id vacancy_id'.split()].rename({'vacancy_id':'predictions'}, axis=1) \
    .explode('predictions') \
    .groupby('user_id') \
    .predictions.agg(lambda x: list(set(x))) \
    .reset_index() \
    .merge(sessions_from_val, on='user_id', how='right')
submission_draft.fillna('none', inplace=True)
submission_draft['predictions'] = submission_draft.apply(lambda row: row.vacancy_id.tolist()+row.predictions if not isinstance(row.predictions, str) else row.vacancy_id , axis=1)
pl.from_pandas(submission_draft['user_id session_id predictions'.split()]).write_parquet('data/submissions/val/train_test_x_to_y.pq')