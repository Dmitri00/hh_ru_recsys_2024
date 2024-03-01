import pandas as pd
from solution.models.recommender_base import UserDataset
class PandasDataFrame(UserDataset):
    def outer_join(self, other, on):
        return pd.merge(self, other, how='outer', on=on)
    
    def inner_join(self, other, on):
        return pd.merge(self, other, how='inner', on=on)
    
    def anti_join(self, other, on):
        return self[~self[on].isin(other[on])]
    
    def cross_join(self, other):
        return pd.merge(self, other)
    
    def rename(self, col_mapping: dict):
        return self.rename(columns=col_mapping)
    
    def sort(self, on, descending=True):
        return self.sort_values(by=on, ascending=not descending)
    
    def select(self, columns):
        return self[columns]
    
    def agg_lists_with_order(self, group_by, list_columns):
        return self.groupby(group_by)[list_columns].agg(list)
    
    def fill_null_column(self, column, fill_value):
        return self.fillna({column: fill_value})
    
    def group_by_and_agg(self, group_by, agg_func: str):
        return self.groupby(group_by).agg(agg_func)
    
    def __len__(self):
        return len(self)

