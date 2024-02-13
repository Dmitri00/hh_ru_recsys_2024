import pandas as pd
from solution.preprocessing.validation import get_test, read_parquet_to_pandas
import sys
import os
class ValidationEval:
    def __init__(self, ground_truth, topk=100):
        self._ground_truth = ground_truth
        self._topk = topk
    def evaluate(self, predicts):
        true_vs_pred = self._ground_truth['user_id target'.split()].merge(predicts['user_id predictions'.split()], on='user_id', how='left')
        true_vs_pred.fillna('none', inplace=True)
        metric = self._metric(true_vs_pred['target'], true_vs_pred['predictions'])
        return metric

class MRRValidation(ValidationEval):
    def _metric(self, true, predicts):
        Q = len(true)
        metric = 0
        for true_item, predicted_items in zip(true, predicts):
            if isinstance(predicted_items, str):
                continue
            metric += self.per_query_metric(true_item, predicted_items)
        return metric / Q
    def per_query_metric(self, true_item, predict_items):
        rank = 1
        found = False
        for predict in predict_items:
            if predict == true_item:
                found = True
                break
            if rank == self._topk:
                break
            rank += 1
        if found :
            return 1 / rank
        else:
            return 0


        
if __name__ == '__main__':
    submission_path = sys.argv[1]
    topk = int(sys.argv[2])
    try:
        row_limit = int(sys.argv[3])
    except:
        row_limit = -1
    test_df = get_test(for_validation=True).to_pandas()
    if row_limit != -1:
        test_df = test_df[:row_limit]
   # topk=100
    validator_at_100 = MRRValidation(test_df, topk)
    
    submission = read_parquet_to_pandas(submission_path)
    score = validator_at_100.evaluate(submission)
    print(f'MRR@{topk} score : {score:.5f}')
    