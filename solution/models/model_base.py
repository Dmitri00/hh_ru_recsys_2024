import sys
from abc import ABC, abstractmethod
from solution.preprocessing.validation import get_train, get_test, store_solution
import logging
#from solution.models.recommender import HistoryIter, Pipeline, RecomPack, PreprocessedDataset, CandidateGenerator


class TrainPredictModelApp(ABC):
        
    @abstractmethod
    def get_args(self):
        pass
    @abstractmethod
    def get_model(self, args):
        pass
    def get_logger(self):
        logging.getLogger().setLevel(logging.INFO)
        return logging.getLogger()

    def main(self):
        args = self.get_args()
        logger = self.get_logger()
        #dataset_fabric = PreprocessedDataset()
        #train = dataset_fabric.get_train(for_validation=args['for_validation'])
        #test = dataset_fabric.get_test(for_validation=args['for_validation'])
        
        model = self.get_model(args)
        
        #model.fit(train)
        
        #queries =  HistoryIter().get_plain_history(test)
        #pipeline = Pipeline()
        #predicts = model.predict(queries)
                
        #store_solution(predicts, args['experiment_name'], for_validation=args['for_validation'])