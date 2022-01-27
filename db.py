import mysql.connector
from mysql.connector import Error
import pandas as pd
from sqlalchemy import create_engine
import datapreparation as dm

# df = dm.return_df()
# df_tripadvisor = dm.return_df_tripadvisor()
# df_booking = dm.return_df_booking()
# df_add = dm.return_df_add()

# dataframes = [df, df_tripadvisor, df_booking, df_add]

# df_final = dm.return_df_final(dataframes)

def upload_data(df):
    engine = create_engine("mysql+mysqlconnector://root:root@localhost:3307/zipcode", 
        connect_args={'connect_timeout': 300})
    try:
        df.to_sql(name='hotelreviews',con=engine, if_exists='replace',index=False,chunksize=1000) 
        print('Succesfully uploaded data')
    except Exception as e:
        print('Something went wrong:', e)

def delete_table(tabla):
    try:
        connection = mysql.connector.connect(
            host="localhost", 
            port=3307,
            user="root", 
            password="root", 
            database="zipcode")
        cursor = connection.cursor()
        cursor.callproc('delete_table', args=(tabla,))
            
        # for result in cursor.stored_results():
        #     results = result.fetchall()
        # df_mysql = pd.DataFrame(results)   

    except mysql.connector.Error as error:
        print("Failed to execute stored procedure: {}".format(error))
    finally:
        if (connection.is_connected()):
            cursor.close()
            connection.close()
            print("MySQL connection is closed")

def clean_db():
    try:
        connection = mysql.connector.connect(
            host="localhost", 
            port=3307,
            user="root", 
            password="root", 
            database="zipcode")
        cursor = connection.cursor()
        cursor.callproc('delete_reviews')
            
        # for result in cursor.stored_results():
        #      results = result.fetchall()
        # df = pd.DataFrame(results)   

    except mysql.connector.Error as error:
        print("Failed to execute stored procedure: {}".format(error))
    finally:
        if (connection.is_connected()):
            cursor.close()
            connection.close()
            print("MySQL connection is closed")
    # return df
def return_db_from_mysql(procedure):
    try:
        connection = mysql.connector.connect(
            host="localhost", 
            port=3307,
            user="root", 
            password="root", 
            database="zipcode")
        cursor = connection.cursor()
        cursor.callproc('delete_reviews')
        cursor.callproc(procedure)
            
        for result in cursor.stored_results():
            results = result.fetchall()
        df = pd.DataFrame(results)
        df = df.rename(columns = {0:'review', 1:'is_positive'})  

    except mysql.connector.Error as error:
        print("Failed to execute stored procedure: {}".format(error))
    finally:
        if (connection.is_connected()):
            cursor.close()
            connection.close()
            print("MySQL connection is closed")

    return df