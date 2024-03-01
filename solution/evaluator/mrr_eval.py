import pandas as pd
from solution.preprocessing.validation import get_test, read_parquet_to_pandas
import sys
import os
import logging
from solution.models.recommender_base import Splits
class ValidationEval:
    def __init__(self, ground_truth, topk=100):
        self._ground_truth = ground_truth
        self._topk = topk
    def evaluate(self, predicts):
        true_vs_pred = self._ground_truth['user_id target'.split()].merge(predicts['user_id predictions'.split()], on='user_id', how='left')
        true_vs_pred.fillna('none', inplace=True)
        metric = self._metric(true_vs_pred['target'], true_vs_pred['predictions'])
        return metric
    
    def _metric(self, true, predicts):
        Q = len(true)
        metric = 0
        for true_item, predicted_items in zip(true, predicts):
            if isinstance(predicted_items, str):
                continue
            metric += self.per_query_metric(true_item, predicted_items)
        return metric / Q

class MRRValidation(ValidationEval):
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

class RecallValidation(ValidationEval):
    def per_query_metric(self, true_item, predict_items):
        found = False
        rank = 1
        for predict in predict_items:
            if predict == true_item:
                found = True
                break
            if rank == self._topk:
                break
            rank += 1
        if found:
            return 1
        else:
            return 0
def parse_split(split):
    if split == 'validation':
        is_validation = Splits('val')
    elif split == 'test':
        is_validation = Splits('test')
    else:
        is_validation = Splits('micro')
    logging.info(f'{is_validation}')
    return is_validation
        
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    submission_path = sys.argv[1]
    topk = int(sys.argv[2])
    try:
        row_limit = int(sys.argv[3])
    except:
        row_limit = -1
    try:
        split = sys.argv[4]
        for_validation = parse_split(split)
    except:
        for_validation = Splits('micro')
    logging.info(f'{for_validation}')
    test_df = get_test(for_validation).to_pandas()
    if row_limit != -1:
        test_df = test_df[:row_limit]
   # topk=100
    
    
    submission = read_parquet_to_pandas(submission_path)
    
    validator_at_100 = MRRValidation(test_df, topk)
    score = validator_at_100.evaluate(submission)
    print(f'MRR@{topk} score : {score:.5f}')
    
    
    validator_at_100 = RecallValidation(test_df, topk)
    score = validator_at_100.evaluate(submission)
    print(f'Recall@{topk} score : {score:.5f}')
    