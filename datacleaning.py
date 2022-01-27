import db
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from dask_ml.linear_model import LogisticRegression as LogisticRegressionDask
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
import lightgbm as lgb
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import recall_score, precision_score, roc_auc_score
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet
import numpy as np
import timeit
import pickle

def cleaning(df):
    
    df["review"] = df["review"].apply(lambda x: x.replace("No Negative", "").replace("No Positive", ""))
    df["review"] = df['review'].str.lower()
    df['review'] = df['review'].str.replace('\d+', '', regex=True)
    df = df.dropna()
    df = df.reset_index(drop=True)

    return df
    
def show_info(df):
    print('\nPercentage of positive and negative reviews: \n',df.is_positive.value_counts()/len(df),'\n')

    length_reviews = df.review.str.len()
    print('\nLongest review: ',max(length_reviews),'\n')
    print('\nShortest review: ',min(length_reviews),'\n')    
    
def display_wordcloud(df):
    my_stopwords = ENGLISH_STOP_WORDS.union(['hotel', 'room', 'staff', 'rooms', 'breakfast', 'bathroom'])

    my_cloud = WordCloud(background_color='white',stopwords=ENGLISH_STOP_WORDS).generate(' '.join(df['review']))
    plt.imshow(my_cloud, interpolation='bilinear') 
    plt.axis("off")
    plt.show()


    df_positive = df[df['is_positive']==1]
    my_cloud_positive = WordCloud(background_color='white',stopwords=my_stopwords).generate(' '.join(df_positive['review']))
    plt.imshow(my_cloud_positive, interpolation='bilinear') 
    plt.axis("off")
    plt.show()

    df_negative = df[df['is_positive']==0]
    my_cloud_negative = WordCloud(background_color='white',stopwords=my_stopwords).generate(' '.join(df_negative['review']))
    plt.imshow(my_cloud_negative, interpolation='bilinear') 
    plt.axis("off")
    plt.show()

def stemming(df):
    # Import the function to perform stemming
    # Call the stemmer
    new_column = []
    porter = PorterStemmer()    # Transform the array of tweets to tokens

    tokens = [word_tokenize(review) for review in df['review']]
    # Stem the list of tokens
    stemmed_tokens = [[porter.stem(word) for word in review] for review in tokens]
    for l in stemmed_tokens:
        stem_sentence = " ".join(l)
        new_column.append(stem_sentence)
        #print(stem_sentence)
    #print(stemmed_tokens)
    df['stem_review'] = new_column
    return df[['stem_review', 'is_positive']]

def pos_tagger(nltk_tag):
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:         
            return None

def lemmatize(df):
    new_column = []
    lemmatizer = WordNetLemmatizer()

    for sentence in df['review']:
        pos_tagged = pos_tag(word_tokenize(sentence)) 
        wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))

        lemmatized_sentence = []
        for word, tag in wordnet_tagged:
            if tag is None:
                lemmatized_sentence.append(word)
            else:       
                lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
        lemmatized_sentence = " ".join(lemmatized_sentence)
        new_column.append(lemmatized_sentence)
    df['lemmatized_review'] = new_column
    return df[['lemmatized_review', 'is_positive']]

def count_vectorizer(df, my_stopwords):
    # Build the vectorizer
    vect = CountVectorizer(stop_words=my_stopwords, ngram_range=(1, 2), 
                             max_features=1000, token_pattern=r'\b[^\d\W][^\d\W]+\b').fit(df.review)
    # Create the bow representation
    X = vect.transform(df.review)
    # Create the data frame
    X_df = pd.DataFrame(X.toarray(), columns=vect.get_feature_names_out())
    return X_df

def tfidf(df, my_stopwords):
    # Build the vectorizer
    vect = TfidfVectorizer(stop_words=my_stopwords, ngram_range=(1, 2), 
                            max_features=1000, token_pattern=r'\b[^\d\W][^\d\W]+\b').fit(df['review'].values.astype('U'))
    # Create sparse matrix from the vectorizer
    X = vect.transform(df['review'].values.astype('U'))
    # Create a DataFrame
    X_df = pd.DataFrame(X.toarray(), columns=vect.get_feature_names_out())
    return X_df

def tfidf_gb(df, my_stopwords):
    # Build the vectorizer
    vect = TfidfVectorizer(stop_words=my_stopwords, ngram_range=(1, 2), 
                            max_features=500, token_pattern=r'\b[^\d\W][^\d\W]+\b').fit(df['review'].values.astype('U'))
    # Create sparse matrix from the vectorizer
    X = vect.transform(df['review'].values.astype('U'))
    # Create a DataFrame
    X_df = pd.DataFrame(X.toarray(), columns=vect.get_feature_names_out())
    return X_df

def first_model(df):
    #Lemmatize, stemming
    my_stopwords = ENGLISH_STOP_WORDS
    df_vect = tfidf(df, my_stopwords)
    # Define X and y
    y = df.is_positive
    X = df_vect

    y_pred_logres = cross_val_predict(LogisticRegression(max_iter=100, penalty='none', random_state=0), X, y, cv=3)
    #Print accuracy score and confusion matrix on test set
    print('Accuracy on the test set: ', accuracy_score(y, y_pred_logres))
    print(y_pred_logres)
    print(confusion_matrix(y, y_pred_logres)/len(y))
    print('Recall on the test set: ',recall_score(y, y_pred_logres))
    print('Precision on the test set: ',precision_score(y, y_pred_logres))
    print('ROC-AUC on the test set: ',roc_auc_score(y, y_pred_logres))

def model_comparator(df, df_vect):
    my_stopwords = ENGLISH_STOP_WORDS

    y = df.is_positive
    X = df_vect

    y_pred_logres = cross_val_predict(LogisticRegression(), X, y, cv=3)
    print('Accuracy for LR: ', accuracy_score(y, y_pred_logres))
    
    y_pred_randomfor = cross_val_predict(RandomForestClassifier(), X, y, cv=3)
    print('Accuracy for RF: ', accuracy_score(y, y_pred_randomfor))

    y_pred_mnb = cross_val_predict(MultinomialNB(), X, y, cv=3)
    print('Accuracy for MNB: ', accuracy_score(y, y_pred_mnb))

    y_pred_gb = cross_val_predict(GradientBoostingClassifier(), X, y, cv=3)
    print('Accuracy GB: ', accuracy_score(y, y_pred_gb))

def models(df):
    my_stopwords = ENGLISH_STOP_WORDS
    df_vect = tfidf(df, my_stopwords)
    df_vect_mnb = count_vectorizer(df, my_stopwords)
    y = df.is_positive
    X = df_vect
    X_mnb = df_vect_mnb
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
    X_train_mnb,X_test_mnb,y_train_mnb,y_test_mnb=train_test_split(X,y,test_size=0.25,random_state=42)

    '''Logistic Regression'''
    
    # parameters_lr = {'max_iter': [100, 150, 200], 'penalty': ['l2', 'none']}
    # lr_model = LogisticRegression(random_state = 0)
    # grid_lr = GridSearchCV(estimator=lr_model, param_grid=parameters_lr, cv=3)
    # grid_lr.fit(X_train, y_train)
    # print("Best parameters for Logistic Regression: ",grid_lr.best_params_)
 
    y_pred_logres = cross_val_predict(LogisticRegression(max_iter=100, penalty='none', random_state=0), X, y, cv=5)
    #Print accuracy score and confusion matrix on test set
    print('Accuracy for Logistic Regression: ', accuracy_score(y, y_pred_logres))
    print(confusion_matrix(y, y_pred_logres)/len(y))
    print('Recall for Logistic Regression: ',recall_score(y, y_pred_logres))
    print('Precision for Logistic Regression: ',precision_score(y, y_pred_logres))
    print('ROC-AUC on the test set: ',roc_auc_score(y, y_pred_logres))

    '''Random Forest Model'''
    # parameters_rf = {'n_estimators': [50, 100, 150],'max_depth': [None,1,2]}
    # rf_model = RandomForestClassifier(random_state = 0)
    # grid_rf = GridSearchCV(estimator=rf_model, param_grid=parameters_rf, cv=3)
    # grid_rf.fit(X_train, y_train)
    # print("Best parameters for Random Forest: ",grid_rf.best_params_)

    start = timeit.timeit()
    y_pred_randomfor = cross_val_predict(RandomForestClassifier(max_depth=None, n_estimators=150, random_state = 0), X, y, cv=5)
    end = timeit.timeit()
    print(end - start)
    #Print accuracy score and confusion matrix on test set
    print('Accuracy for Random Forest Classifier: ', accuracy_score(y, y_pred_randomfor))
    print(confusion_matrix(y, y_pred_randomfor)/len(y))
    print('Recall for Random Forest: ',recall_score(y, y_pred_randomfor))
    print('Precision for Random Forest: ',precision_score(y, y_pred_randomfor))
    print('ROC-AUC on the test set: ',roc_auc_score(y, y_pred_randomfor))


    '''Multinomial Naive Bayes Model'''
    # parameters_mnb = {'fit_prior': [True, False],'alpha': [0, 0.1, 1]}
    # mnb_model = MultinomialNB()
    # grid_mnb = GridSearchCV(estimator=mnb_model, param_grid=parameters_mnb, cv=3)
    # grid_mnb.fit(X_train, y_train)
    # print("Best parameters for Multinomial Naive Bayes: ",grid_mnb.best_params_)

    start = timeit.timeit()
    y_pred_mnb = cross_val_predict(MultinomialNB(alpha=0, fit_prior=False), X_mnb, y, cv=5)
    end = timeit.timeit()
    #Print accuracy score and confusion matrix on test set
    print('Accuracy on the test set for Multinomial Naive Bayes: ', accuracy_score(y, y_pred_mnb))
    print(confusion_matrix(y, y_pred_mnb)/len(y))
    print('Recall for Multinomial Naive Bayes Model: ',recall_score(y, y_pred_mnb))
    print('Precision for Multinomial Naive Bayes Model: ',precision_score(y, y_pred_mnb))
    print('ROC-AUC on the test set: ',roc_auc_score(y, y_pred_mnb))
 
    '''Gradient Boost Model'''
    # parameters_gb = {'n_estimators': [50, 100, 200],'learning_rate': [0.001, 0.01, 0.1],}
    # gb_model = GradientBoostingClassifier(random_state = 0)
    # grid_gb = GridSearchCV(estimator=gb_model, param_grid=parameters_gb, cv=3)
    # grid_gb.fit(X_train, y_train)
    # print("Best parameters for GB: ",grid_gb.best_params_)

    df_vect_gb = tfidf_gb(df, my_stopwords)
    X_gb = df_vect_gb

    start = timeit.timeit()
    y_pred_gb = cross_val_predict(GradientBoostingClassifier(learning_rate=0.1, n_estimators=200, random_state=0), X_gb, y, cv=5)
    end = timeit.timeit()
    print(end - start)
    #Print accuracy score and confusion matrix on test set
    print('Accuracy on the test set for Gradient Boosting Classifier: ', accuracy_score(y, y_pred_gb))
    print(confusion_matrix(y, y_pred_gb)/len(y))
    print('Recall for Gradient Boosting Classifier: ',recall_score(y, y_pred_gb))
    print('Precision for Gradient Boosting Classifier: ',precision_score(y, y_pred_gb))
    print('ROC-AUC on the test set: ',roc_auc_score(y, y_pred_gb))

def save_models(df):
    my_stopwords = ENGLISH_STOP_WORDS
    df_vect = tfidf(df, my_stopwords)
    #df_vect_mnb = count_vectorizer(df, my_stopwords)
    y = df.is_positive
    X = df_vect
    #X_mnb = df_vect_mnb

    print('LR Dask training...')
    model_lr = LogisticRegressionDask(max_iter=100, penalty='none', random_state=0, fit_intercept=False)
    model_lr.fit(X, y)
    filename_lr = 'model_lr_dask.sav'
    pickle.dump(model_lr, open(filename_lr, 'wb'))

    # print('LR training...')
    # model_lr = LogisticRegression(max_iter=100, penalty='none', random_state=0)
    # model_lr.fit(X, y)
    # filename_lr = 'model_lr.sav'
    # pickle.dump(model_lr, open(filename_lr, 'wb'))

    # print('RF training...')
    # model_rf = RandomForestClassifier(max_depth=None, n_estimators=150, random_state = 0)
    # model_rf.fit(X, y)
    # filename_rf = 'model_rf.sav'
    # pickle.dump(model_rf, open(filename_rf, 'wb'))

    # print('MNB training...')

    # model_mnb = MultinomialNB(alpha=0, fit_prior=False)
    # model_mnb.fit(X_mnb, y)
    # filename_mnb = 'model_mnb.sav'
    # pickle.dump(model_mnb, open(filename_mnb, 'wb'))

    # print('GB training...')

    # model_gb = GradientBoostingClassifier(learning_rate=0.1, n_estimators=200, random_state=0)
    # model_gb.fit(X, y)
    # filename_gb = 'model_gb.sav'
    # pickle.dump(model_gb, open(filename_gb, 'wb'))

def save_vectorizers(df):
    print('Saving count vectorizer...\n')

    count_vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS, ngram_range=(1, 2), lowercase=True,
                             max_features=1000, token_pattern=r'\b[^\d\W][^\d\W]+\b').fit(df.review)
    count_vectorizer.fit(df.review)
    pickle.dump(count_vectorizer, open("count_vectorizer.pickel", "wb"))

    print('Saving tfidf...\n')

    tfidf = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS, ngram_range=(1, 2), lowercase=True,
                            max_features=1000, token_pattern=r'\b[^\d\W][^\d\W]+\b').fit(df['review'].values.astype('U'))
    tfidf.fit(df['review'].values.astype('U'))
    pickle.dump(tfidf, open("tfidf.pickel", "wb"))

    print('Saving tfidf_gb...\n')

    tfidf_gb = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS, ngram_range=(1, 2), lowercase=True,
                            max_features=500, token_pattern=r'\b[^\d\W][^\d\W]+\b').fit(df['review'].values.astype('U'))
    tfidf_gb.fit(df['review'].values.astype('U'))
    pickle.dump(tfidf_gb, open("tfidf_gb.pickel", "wb"))

def assesment():
    data = [['I really dislike this hotel, the sheets where dirty and the smell was so bad', 0], 
    ['I think this is a good hotel, the staff was nice and the breakfast delicious', 1], 
    ['It was a good experience to be here, the room was very clear and very quiet, I like it', 1]]
  
    df = pd.DataFrame(data, columns = ['review', 'is_positive'])
    df = cleaning(df)
    df = stemming(df)
    df = df.rename(columns = {'stem_review':'review'})
    #print(df.head())

    cv = pickle.load(open("count_vectorizer.pickel", "rb"))
    X_cv = cv.transform(df.review)
    X_df_cv = pd.DataFrame(X_cv.toarray(), columns=cv.get_feature_names_out())
    #print(X_df_cv.head())

    tfidf = pickle.load(open("tfidf.pickel", "rb"))
    X_tfidf = tfidf.transform(df['review'].values.astype('U'))
    X_df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=tfidf.get_feature_names_out())
    #print(X_df_tfidf.head())

    tfidf_gb = pickle.load(open("tfidf_gb.pickel", "rb"))
    X_tfidf_gb = tfidf_gb.transform(df['review'].values.astype('U'))
    X_df_tfidf_gb = pd.DataFrame(X_tfidf_gb.toarray(), columns=tfidf_gb.get_feature_names_out())
    #print(X_df_tfidf_gb.head())

    lr = pickle.load(open('model_lr.sav', 'rb'))
    rf = pickle.load(open('model_rf.sav', 'rb'))
    mnb = pickle.load(open('model_mnb.sav', 'rb'))
    gb = pickle.load(open('model_gb.sav', 'rb'))

    y_lr = lr.predict(X_df_tfidf)
    prob_lr = lr.predict_proba(X_df_tfidf)

    y_rf = rf.predict(X_df_tfidf)
    prob_rf = rf.predict_proba(X_df_tfidf)


    y_mnb = mnb.predict(X_df_cv)
    prob_mnb = mnb.predict_proba(X_df_cv)

    y_gb = gb.predict(X_df_tfidf)
    prob_gb = gb.predict_proba(X_df_tfidf)


    print(df.head())
    print('\nLogistic Regression predicts: ', y_lr, prob_lr)
    print('\nRandom Forest predicts: ', y_rf, prob_rf)
    print('\nMultinomial Naive Bayes predicts: ', y_mnb, prob_mnb)
    print('ºnGradient Boosting predicts: ', y_gb, prob_gb)

