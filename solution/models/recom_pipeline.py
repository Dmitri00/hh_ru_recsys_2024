from solution.preprocessing.validation import get_test, get_train
from solution.models.continue_search import AllXToSubmit
from solution.models.recommender_base import *
from solution.models.common_stages import ConstantRanker, CatBoostLikeRanker, HistoryIter, ConstantCandidateGenerator


class SimpleDatasetFabric:
    user_history = pl.DataFrame(
            {'user_id':[1,2, 3],
            'session_id':['1','2', '3'],
            'vacancy_id': [[1,2], [1,2], [1,2,3,4]],
            'action_type':[[1,2], [2,1], [1,2,2,1]],
            'action_dt': [['1','2'], ['1','2'], ['1','1','1','1']],
        })
    def get_train(self, split):
        return self.user_history
    def get_test(self, split):
        return self.user_history

class HistoryFilterClicked(HistoryFilter):
    def __init__(self, user_history):
        self._user_history = user_history.filter(pl.col('action_type') == 1)

logging.getLogger().setLevel(logging.INFO)



dataset_fabric = SimpleDatasetFabric()

history_iter = HistoryIter()

user_history = SimpleDatasetFabric().get_test('test')

user_history_train = SimpleDatasetFabric().get_train('test')

plain_history_test = history_iter.get_plain_history(user_history)



some_candidates = pl.DataFrame({'vacancy_id':[1,2,3,4], 'score':[1,2,3,4]})

candgen_als = ConstantCandidateGenerator(some_candidates, 'als')
candgen_text = ConstantCandidateGenerator(some_candidates, 'knn_text')

joint_candidates = JoinCandidatesStage([candgen_als, candgen_text, AllXToSubmit('all_x_to_y')], join_by='user_id vacancy_id'.split())

shown_filter = HistoryFilterClicked(history_iter.get_plain_history(user_history_train))

reranker = CatBoostLikeRanker('constant_ranker', 'vacancy_id'.split())

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

pipeline.predict(plain_history_test)
