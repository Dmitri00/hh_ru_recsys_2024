best_predict = validation_sessions['user_id vacancy_id'.split()].copy()
best_predict['predictions'] = best_predict['vacancy_id']#.apply(lambda x: [x])
pl.from_pandas(best_predict).write_parquet('data/submissions/val/test_x_to_y.pq')
#validator.evaluate(best_predict)