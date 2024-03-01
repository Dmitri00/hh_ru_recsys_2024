import sys
from abc import ABC, abstractmethod
from solution.preprocessing.validation import store_solution, UserLogDataset
import logging
from solution.models.recommender_base import *

from solution.models.common_stages import ConstantRanker, CatBoostLikeRanker, HistoryIter
from solution.models.common_stages import I2IListModel, HistoryFilterClicked, CandGenWrapper
    


class HistoryFilterClickedOnlyOtcliks(HistoryFilterClicked):
    def __init__(self, user_history: UserDataset):
        super().__init__(user_history.filter(pl.col('action_type') == 1))

class TrainPredictModelApp(ABC):
        
    @abstractmethod
    def get_args(self):
        pass
    @staticmethod    
    def parse_split(split):
        if split == 'validation':
            is_validation = Splits('val')
        elif split == 'test':
            is_validation = Splits('test')
        else:
            is_validation = Splits('micro')
        logging.info(f'{is_validation}')
        return is_validation
    @abstractmethod
    def get_model(self, args):
        pass
    def after_submit(self, args):
        pass
    def get_logger(self):
        logging.getLogger().setLevel(logging.INFO)
        return logging.getLogger()
    
    def main(self):
        ## Этот модуль поломан, тк модель продолжить просмотр теперь принимает explode историю
        # и возвращает незапакованные, плоские рекомы
        
        args = self.get_args()
        logger = self.get_logger()
        PipelineInfo.set_split(args['for_validation'])
        logging.info(f'{PipelineInfo.SPLIT}')

        dataset_fabric = UserLogDataset()

        history_iter = HistoryIter()
    
        user_history = dataset_fabric.get_test()
    
        user_history_train = dataset_fabric.get_train()
        plain_history_train = history_iter.get_plain_history(user_history_train)

        
        # self.pipeline.add_stage(submitter)
    
        model = self.get_model(args)

        joint_candidates = JoinCandidatesStage([model], join_by='user_id vacancy_id'.split())

        shown_filter = HistoryFilterClicked(plain_history_train)

        # задание имени тоже должно переехать в отдельный метод
        reranker = ConstantRanker('constant_ranker')

        recom_pack = RecomPack('packer')

        pipeline = ListPipeline(
        ) 
        pipeline.add_stage(joint_candidates)
        pipeline.add_stage(shown_filter)
        pipeline.add_stage(reranker)

        pipeline.add_stage(recom_pack)
    
        #pipeline.fit(plain_history_train)
    
        logging.info(f'Test dataset {user_history}')
        plain_history_test = history_iter.get_plain_history(user_history)
        logging.info(f'Test dataset plain {plain_history_test}')
    
        predict = pipeline.predict(plain_history_test)
    
        submitter = SubmitPrepare(user_history)
    
        submission = submitter.predict(predict)
    
        store_solution(submission, args['experiment_name'], PipelineInfo.SPLIT)

        self.after_submit(args)
        
        #store_solution(predicts, args['experiment_name'], for_validation=args['for_validation'])