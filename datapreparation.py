# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 15:30:06 2021

@author: Jaime
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

def return_df():
    df = pd.read_csv('Hotel_Reviews.csv', header=0)
    df = df.drop(columns =['Hotel_Address',	'Additional_Number_of_Scoring',	'Review_Date',	'Average_Score',	
                  'Hotel_Name',	'Reviewer_Nationality',	'Review_Total_Negative_Word_Counts',	
                  'Total_Number_of_Reviews',	'Review_Total_Positive_Word_Counts',	
                  'Total_Number_of_Reviews_Reviewer_Has_Given',	'Reviewer_Score',	'Tags',	'days_since_review',	
                  'lat',	'lng'])
    df = df.melt(value_vars=['Positive_Review', 'Negative_Review'], var_name = 'rating', value_name = 'review')
    df["rating"].replace({"Positive_Review": 1, "Negative_Review": 0}, inplace=True)
    
    df = df[["review", "rating"]]
    return df

def first_insight():
    df = pd.read_csv('Hotel_Reviews.csv', header=0)

    reviews_per_hotel = df.Hotel_Name.value_counts()
    #reviews_per_hotel.to_csv('reviews_per_hotel.csv', index=True, header=True)
    print(reviews_per_hotel)
    reviews_per_hotel[0:10].plot(kind='bar')
    plt.show()

    reviews_per_nationality = df.Reviewer_Nationality.value_counts()
    #reviews_per_nationality.to_csv('reviews_per_nationality.csv', index=True, header=True)
    print(reviews_per_nationality)
    reviews_per_nationality[0:10].plot(kind='bar')
    plt.show()

    print(df.info())
    # Añadir número de palabras
    df["word_len_pos"] = df["Positive_Review"].apply(lambda x: len(x.split(" ")))
    df["word_len_neg"] = df["Negative_Review"].apply(lambda x: len(x.split(" ")))
    fig = plt.figure(figsize=(10,6))
    plt_neg = sns.distplot(df.word_len_neg, hist=True)
    plt_pos = sns.distplot(df.word_len_pos, hist=True)
    fig.legend(labels=['Negative review','Positive review'])
    plt.show()

    
    
    print('Tripadvisor:')
    df_tripadvisor = return_df_tripadvisor()
    print(df_tripadvisor.info())
    reviews_per_posneg = df_tripadvisor.rating.value_counts()
    reviews_per_posneg.plot(kind='bar')
    plt.show()
    print('Booking:')
    df_booking = return_df_booking()
    print(df_booking.info())
    reviews_per_posneg = df_booking.rating.value_counts()
    reviews_per_posneg.plot(kind='bar')
    plt.show()

def return_df_tripadvisor():
    #Tripadvisor:
    df_tripadvisor = pd.read_csv('reviewtripadvisor.csv', sep=";")
    df_tripadvisor.loc[df_tripadvisor.rating >= 30, 'rating'] = 1
    df_tripadvisor.loc[df_tripadvisor.rating != 1, 'rating'] = 0
    return df_tripadvisor

def return_df_booking():
    #Booking:
    df_booking = pd.read_csv('reviewbooking.csv', sep=";")
    df_booking["rating"].replace({"Liked": 1, "Disliked": 0}, inplace=True)
    return df_booking

def return_df_add():
    #Writehanded reviews
    d = {'review': ['I dont like this hotel, the sheets were dirty and the staff was rude', 'I like this hotel, I slept very well and everything was clean'], 'rating': [0, 1]}
    df_add = pd.DataFrame(data=d)
    return df_add

def return_df_final():
    df = return_df()
    df_tripadvisor = return_df_tripadvisor()
    df_booking = return_df_booking()
    df_add = return_df_add()
    dataframes = [df, df_tripadvisor, df_booking, df_add]

    df_final = pd.concat(dataframes)
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    df_final = df_final.rename(columns={"rating": "is_positive"})
    return df_final

def return_stem_df():
    stem_df = pd.read_csv('stem_df.csv', sep=";")
    stem_df = stem_df.rename(columns = {'stem_review':'review'})
    stem_df = stem_df.dropna()
    stem_df = stem_df.reset_index(drop=True)
    return stem_df

def return_lemmatized_df():
    lemmatized_df = pd.read_csv('lemmatized_df.csv', sep=";")
    lemmatized_df = lemmatized_df.rename(columns = {'lemmatized_review':'review'})
    lemmatized_df = lemmatized_df.dropna()
    lemmatized_df = lemmatized_df.reset_index(drop=True)
    return lemmatized_df

# df = return_df()
# df_tripadvisor = return_df_tripadvisor()
# df_booking = return_df_booking()
# df_add = return_df_add()

# dataframes = [df, df_tripadvisor, df_booking, df_add]

# df_final = return_df_final(dataframes)

# print(df.head())
# print(df.info())


# print(df_tripadvisor.head())
# print(df_tripadvisor.info())


# print(df_booking.head())
# print(df_booking.info())

# print(df_add.head())
# print(df_add.info())


# print(df_final.head())
# print(df_final.info())


