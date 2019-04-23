import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer


def main():
    print('Read training data ...')
    train_df = pd.read_csv('/home/work/Workspaces/deepshare-cs224n-homework'
                           '/data/text_classification_data/train_set.csv')

    print('Read testing data ...')
    test_df = pd.read_csv('/home/work/Workspaces/deepshare-cs224n-homework'
                          '/data/text_classification_data/test_set.csv')

    print('Drop article, id from training dataframe ...')
    train_df.drop(columns=['article', 'id'], inplace=True)
    print('Drop article from testing dataframe ...')
    test_df.drop(columns=['article'], inplace=True)

    print('Vectorizing ...')
    vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, max_features=100)
    vectorizer.fit(train_df['word_seg'])
    print('Transform training data ...')
    X_train = vectorizer.transform(train_df['word_seg'])
    print('Transform testing data ...')
    X_test = vectorizer.transform(test_df['word_seg'])
    y_train = train_df['class'] - 1

    print('Training ...')
    lg = LogisticRegression(C=4, dual=True)
    lg.fit(X_train, y_train)

    print('Predicting ...')
    y_test = lg.predict(X_test)

    print('Save result as local file ...')
    test_df['class'] = y_test.tolist()
    test_df['class'] = test_df['class'] + 1

    result_df = test_df.loc[:, ['id', 'class']]
    result_df.to_csv('result.csv')

    print('Done!')


if __name__ == '__main__':
    main()
    result_df = pd.read_csv('result.csv')
    save_df = result_df[['id', 'class']]
    save_df.to_csv('result0.csv', index=False)
    print('ok')
