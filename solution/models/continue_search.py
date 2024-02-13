
from solution.preprocessing.validation import get_train, get_test, store_solution

class TextXToSubmit:
    def fit(self, df):
        pass
    def predict(self, df):
        return df['user_id vacancy_id session_id'.split()].rename({'vacancy_id':'predictions'}, axis=1)

if __name__ == '__main__':
    model = TextXToSubmit()
    train = get_train(for_validation=True)
    test = get_test(for_validation=True)
    
    model.fit(train)
    
    predicts = model.predict(test)
    
    store_solution(predicts, 'test_x_to_submit', for_validation=True)

    
