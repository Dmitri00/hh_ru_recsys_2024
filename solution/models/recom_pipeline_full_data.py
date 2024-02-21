from re import sub
from solution.preprocessing.validation import get_test, get_train, store_solution
from solution.models.continue_search import AllXToSubmit
from solution.models.recommender_base import *

from solution.models.common_stages import ConstantRanker, CatBoostLikeRanker, HistoryIter



class PreProcessedDataset(DatasetFabric):
    def get_train(self, split: Splits):
        return get_train(self.parse_split(split))
    def get_test(self, split: Splits):
        return get_test(self.parse_split(split))
    def parse_split(self, split: Splits):
        if split == Splits.VALIDATION:
            is_validation = True
        else:
            is_validation = False
        return is_validation

class HistoryFilterClicked(HistoryFilter):
    def __init__(self, user_history):
        self._user_history = user_history.filter(pl.col('action_type') == 1)


logging.getLogger().setLevel(logging.INFO)



dataset_fabric = PreProcessedDataset()

history_iter = HistoryIter()

user_history = dataset_fabric.get_test(Splits('val'))

user_history_train = dataset_fabric.get_train(Splits('val'))



joint_candidates = JoinCandidatesStage([AllXToSubmit('all_x_to_y')], join_by='user_id vacancy_id'.split())

shown_filter = HistoryFilterClicked(history_iter.get_plain_history(user_history_train))

reranker = ConstantRanker('constant_ranker')

recom_pack = RecomPack('packer')

submitter = SubmitPrepare(user_history)

pipeline = ListPipeline(
) 
pipeline.add_stage(joint_candidates)
pipeline.add_stage(shown_filter)
pipeline.add_stage(reranker)

pipeline.add_stage(recom_pack)

pipeline.add_stage(submitter)

pipeline.fit(history_iter.get_plain_history(user_history_train))

logging.info(f'Test dataset {user_history}')
plain_history_test = history_iter.get_plain_history(user_history)
logging.info(f'Test dataset plain {plain_history_test}')

submission = pipeline.predict(plain_history_test)

store_solution(submission, 'recom_pipeline_continue_search', True)

