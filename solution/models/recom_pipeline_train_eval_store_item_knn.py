from re import sub
from solution.preprocessing.validation import get_test, get_train, store_solution, get_vacancies
from solution.models.continue_search import AllXToSubmit
from solution.models.recommender_base import *

from solution.models.common_stages import ConstantRanker, CatBoostLikeRanker, HistoryIter
from solution.models.common_stages import I2IListModel

from sklearn.preprocessing import LabelEncoder

from solution.models.knn import get_item_geo_i2i_pca_fast_faiss_knn



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
    def fit(self, user_history: UserDataset):
        self._user_history = user_history.filter(pl.col('action_type') == 1)


class PolarsI2ICandidateGenerator(I2IListModel):
    def __init__(self, i2i_dataframe):
        self._i2i = i2i_dataframe

    
class RecomPipelineApp:
    def __init__(self):
        item_knn_stage = AllXToSubmitStage().set_name('all_x_to_y')
        joint_candidates = JoinCandidatesStage([item_knn_stage], join_by='user_id vacancy_id'.split())

        shown_filter = HistoryFilterClicked()

        # задание имени тоже должно переехать в отдельный метод
        reranker = ConstantRanker('constant_ranker')

        recom_pack = RecomPack('packer')

        

        self.pipeline = ListPipeline(
        ) 
        self.pipeline.add_stage(joint_candidates)
        self.pipeline.add_stage(shown_filter)
        self.pipeline.add_stage(reranker)

        self.pipeline.add_stage(recom_pack)



    def fit(self, x):
        return self.pipeline.fit(x)
    def predict(self, x):
        return self.pipeline.predict(x)



def main():
    logging.getLogger().setLevel(logging.INFO)
    dataset_fabric = PreProcessedDataset()

    history_iter = HistoryIter()

    user_history = dataset_fabric.get_test(Splits('val'))

    user_history_train = dataset_fabric.get_train(Splits('val'))

    
    # self.pipeline.add_stage(submitter)

    pipeline = RecomPipelineApp()

    plain_history_train = history_iter.get_plain_history(user_history_train)

    
    
    pipeline.fit(plain_history_train)

    logging.info(f'Test dataset {user_history}')
    plain_history_test = history_iter.get_plain_history(user_history)
    logging.info(f'Test dataset plain {plain_history_test}')

    predict = pipeline.predict(plain_history_test)

    submitter = SubmitPrepare(user_history)

    submission = submitter.predict(predict)

    store_solution(submission, 'recom_pipeline_continue_search', True)

if __name__ == '__main__':
    main()