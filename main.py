import db
import datapreparation as dp
import datacleaning as dc
import warnings # Control de advertencias
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
warnings.filterwarnings('ignore')
if __name__ == "__main__":
    # RUN CODE HERE
    
    # df_final = dp.return_df_final()
    # # print(df_final.head())
    # db.upload_data(df_final)
    # # db.clean_db()
    # df = db.return_db_from_mysql()
    # print(df.info())
    # df = dc.cleaning(df)
    # df = dc.stemming(df)
    # print(df.head())  
    # print(df.info())
    # #df = dc.lemmatize(df)
    # df.to_csv("stem_df.csv", sep=';',index= False)
    
    # print(df.info())
    # print(df.head())
    #df_lemmatized = dp.return_lemmatized_df()
    #dc.display_wordcloud(df_lemmatized)
    # dc.first_model(df)
        # RUN CODE HERE

    #stem_df = dp.return_stem_df()
    # print(stem_df.info())
    # lemmatized_df = dp.return_lemmatized_df()
    # print(lemmatized_df.info())

    #Stemming:
    #Accuracy on the test set:  0.893849129777555
    # [[0.39047618 0.04486303]
    #  [0.06128784 0.50337295]]
    stem_df = db.return_db_from_mysql('select_all_stem')
    print('We have the dataaaa')
    
    dc.save_models(stem_df)
    # print(stem_df.info())
    #dc.show_info(stem_df)
    #dp.first_insight()
    #dc.save_vectorizers(stem_df)
    #dc.assesment()
    #Lemmatize
    #Accuracy on the test set:  0.884887447588943
    #[[0.38700883 0.04833039]
    #[0.06678217 0.49787862]]
    #lemmatized_df = db.return_db_from_mysql('select_all_lemmatized')
    # print(lemmatized_df.info())
    #dc.first_model(lemmatized_df)

    #dp.first_insight()
    
    ##TEST:
    # my_stopwords = ENGLISH_STOP_WORDS.union(['hotel', 'room', 'staff', 'rooms', 'breakfast', 'bathroom'])
    # df_stem = dp.return_stem_df()
    # df_stem = df_stem.head(100000)
    # df_stem_CV = dc.count_vectorizer(df_stem, my_stopwords)
    # df_stem_TFIDF = dc.tfidf(df_stem, my_stopwords)

    # df_lemmatized = dp.return_lemmatized_df()
    # df_lemmatized = df_lemmatized.head(100000)
    # df_lemmatized_CV = dc.count_vectorizer(df_lemmatized, my_stopwords)
    # df_lemmatized_TDIDF = dc.tfidf(df_lemmatized, my_stopwords)

    # print('Test with stemming and count vectorizer:')
    # dc.model_comparator(df_stem, df_stem_CV)
    # print('Test with lemmatizing and count vectorizer:')
    # dc.model_comparator(df_lemmatized, df_lemmatized_CV)

    # print('Test with stemming and tfidf:')
    # dc.model_comparator(df_stem, df_stem_TFIDF)
    # print('Test with lemmatizing and tfidf:')
    # dc.model_comparator(df_lemmatized, df_lemmatized_TDIDF)

    ############################################################
    #ASSESMENT
    
    dc.assesment()