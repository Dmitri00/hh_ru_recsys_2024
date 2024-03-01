
import polars as pl
import logging
from solution.models.model_base import TrainPredictModelApp, Splits
import sys
from solution.models.recommender_base import CandidateGenerator, PipelineStage
from solution.models.common_stages import CandGenWrapper
import sys
from solution.preprocessing.validation import UserLogDataset

class AllXToSubmit:
    def __init__(self, action_weights=None):
        if action_weights is not None:
            self._action_weights = action_weights
        else:
            self._action_weights = {1:-5, 2:1, 3:3}
    def _preproc_sessions(self, user_session):
        #user_items = user_session.explode('vacancy_id', 'action_type', 'action_dt')
        user_items = user_session
        #logging.info(f'{user_session}')
        user_items = user_items.with_columns(
            action_weight=user_items['action_type'].map_elements(self._remap_actions)
        )
        user_items = user_items.select(
            'user_id', 'vacancy_id', 'action_type',
            'action_weight',
            'action_dt'
        )
        logging.info(f'Колонки в распаршеной сессии {user_items.columns}')

        return user_items
        
    def _remap_actions(self, action_weight):
        return self._action_weights[action_weight]
    
    def _aggregate_actions(self, user_item_interactions):
        user_item_interactions = user_item_interactions.sort('action_weight', descending=True)
        user_items = user_item_interactions \
            .group_by('user_id', 'vacancy_id', maintain_order=True) \
            .first()
        
        ## если просто отфильтровать здесь отклики, то у юзеров с одними отклами в тесте без истории в трейне не будет рекомов
        #user_items = user_items.filter(pl.col('action_type') != 1)
        
        
        return user_items
            
        
    def fit(self, df):        
        # explode каждой сессии пользователя
        # ремап действий юзера в веса - точно нужно, тк нужна соритровка, а ид событий не упорядочены по бизнес логике
        logging.info(f'кол-во сессий {len(df)}')
        logging.info('препроц юзер сессий в трейне')
        user_items = self._preproc_sessions(df)
        logging.info('препроц юзер сессий в трейне завершен')
        logging.info(f'кол-во айтемов {len(user_items)}')
        
        # groupby действий пользователя с макс семплом действия по айтему
        # убрать айтемы, у которых был отклик по максемплу
        self._user_items_aggregated = self._aggregate_actions(user_items)

                
        return self
    
    def _sort_recoms(self, user_recoms):
        # тут нужно разработать скор например (-log(position)+action_weight)
        # map_column
        user_recoms = user_recoms.with_columns(postion_rank=pl.col('action_dt').rank(method='dense', descending=True).over('user_id'))
        user_recoms = user_recoms.with_columns(rank=pl.col('postion_rank').log())
        return user_recoms.sort('rank')
    
    def _pack_recoms(self, user_row_recoms):
        return user_row_recoms.group_by('user_id', maintain_order=True) \
            .agg(pl.col('vacancy_id').alias('predictions'))
    
    def predict(self, df):
        # препроц сессий:
        # explode сессии каждого пользователя
        # ремап действий в веса
        logging.info(f'Кол-во юзер сессий для предикта {len(df)}')
        predict_users = df['user_id'].unique()
        user_items = self._preproc_sessions(df)
                             
        # объединить тестовую сессию с отранжированными действиями в трейне
        user_items_aggregated = self._user_items_aggregated.filter(
                pl.col('user_id').is_in(predict_users))
        all_user_interactions = pl.concat([user_items_aggregated, user_items])
        # повторить шаги аггрегации действий:
        # groupby действий пользователя с макс семплом действия по айтему
        # убрать айтемы, у которых был отклик по максемплу
        aggregated_user_items = self._aggregate_actions(all_user_interactions)
        # cортировка действий
        # сортировать айтему по макс семплу, потом таймстемпу по убыванию
        
        sorted_user_recoms = self._sort_recoms(aggregated_user_items)
        # упаковать построчные рекомы в списки
        
        #predictions = self._pack_recoms(sorted_user_recoms)
        
        #predictions = df.select('user_id session_id'.split()).join(predictions, on='user_id')
        #logging.info(f'Кол-во юзеров в предикте {len(predictions)}')

        
        return sorted_user_recoms.rename({'rank':'score'}).select('user_id', 'vacancy_id', 'score')


class AllXToSubmitApp(TrainPredictModelApp):
    def __init__(self):
        self._model = AllXToSubmit()

    
    def get_args(self):
        try:
            split = sys.argv[1]
            for_validation = AllXToSubmitApp.parse_split(split)
        except:
            for_validation = Splits('val')
        return {'for_validation': for_validation,
               'experiment_name':'all_x_to_y_sort_by_rank'}
    def get_model(self, args):
        train = UserLogDataset().get_train() \
            .explode('action_type', 'vacancy_id', 'action_dt')
        
        return CandGenWrapper(self._model.fit(train)).set_name('continue_search')
        
        
if __name__ == '__main__':
    
    AllXToSubmitApp().main()
    

    
