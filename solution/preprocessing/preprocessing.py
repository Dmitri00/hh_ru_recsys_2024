import pandas as pd
class IdParser:
    def __init__(self, original_col, parsed_col, prefix):
        self.original_col = original_col
        self.parsed_col = parsed_col
        self.prefix = prefix
    def transform(self, df):
        df[self.original_col] = df[self.original_col].apply(self.id_parser)
        return df
    def inverse_transform(self, df):
        df[self.original_col] = df[self.original_col].apply(self.id_construct)
        return df
    def id_parser(self, user_id_str):
        return int(user_id_str[len(self.prefix):])
    def id_construct(self, id_):
        return f'{self.prefix}{id_}'

class IdArrayParser(IdParser):
    def id_parser(self, id_arr):
        return [super(IdArrayParser, self).id_parser(id_) for id_ in id_arr] 

    def id_construct(self, id_arr):
        return [super(IdArrayParser, self).id_construct(id_) for id_ in id_arr] 


class Pipeline:
    def __init__(self, stages):
        self.stages = stages
    def transform(self, x):
        for stage in self.stages:
            x = stage.transform(x)
        return x
    def inverse_transform(self, x):
        for stage in reversed(self.stages):
            x = stage.inverse_transform(x)
        return x



class SessionDt:
    def __init__(self, dt_arr_col):
        self.dt_col = dt_arr_col
    def transform(self, df):
        df['session_dt'] = df[self.dt_col].apply(lambda dts: dts[0])
        return df
    def inverse_transform(self, df):
        return df

class ActionDtReorder:
    def __init__(self, order_col, columns_to_reorder):
        self._order_col = order_col
        self._columns_to_reorder = columns_to_reorder
    def _sort_dt_session_actions(self, row):
        session_dt_order = row[self._order_col].argsort()
        for field_name in [self._order_col] + self._columns_to_reorder:
            row[field_name] = row[field_name][session_dt_order]
        return row
    def transform(self, df):
        return df.apply(self._sort_dt_session_actions, axis=1)


base_preprocessing = Pipeline([
    ActionDtReorder('action_dt', 'vacancy_id action_type'.split()),
    SessionDt('action_dt'),
])