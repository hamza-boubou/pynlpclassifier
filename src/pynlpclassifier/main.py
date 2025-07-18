import gensim
import pandas as pd
import os
import Levenshtein
import numpy as np

from pyspark.sql import Window
import re

from pyspark.sql.types import StringType, IntegerType, ArrayType
from pyspark.broadcast import Broadcast

# from pyspark.sql import SparkSession
from . import utils as fx
from pyspark.sql import functions as F

stopwords_list = [
    'de', 'la', 'el', 'y', 'en', 'con', 'del', 'los', 'las', 'para', 'por', 'a',
    'un', 'una', 'al', 'o', 'u', 'e', 'se', 'su', 'que', 'como',
    'the', 'of', 'on', 'in', 'by', 'with', 'that', 'and', 'for'
]

accents_list = [
    ("á", "a"),
    ("é", "e"),
    ("í", "i"),
    ("ó", "o"),
    ("ú", "u"),
    ("ñ", "n")
]

def train_ml_model(
    df,
    text_column,
    model_path="trained_model.model",
    test_word = "industrial",
    window=7,
    min_count=2,
    workers=4
):
    """
    Trains a Word2Vec model on a specified text column of a DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing text data
    - text_column (str): Name of the column with text to train on
    - model_path (str): Path to save the trained model

    Returns:
    - model (gensim.models.Word2Vec): Trained Word2Vec model
    """

    # Clean the text column
    df[text_column] = df[text_column].apply(fx.clean_text)

    # Tokenize: Word2Vec expects a list of lists of tokens
    tokenized_text = df[text_column].apply(str.split).tolist()
    
    # Initialize model
    model = gensim.models.Word2Vec(
        sentences=tokenized_text,
        window=window,
        min_count=min_count,
        workers=workers
    )
    
    # Save model
    model.save(model_path)
    
    # Test the model
    try:
        print(model.wv.most_similar(test_word))
    except KeyError:
        print(f"⚠️ The word '{test_word}' is not in the model vocabulary.")


""" spark functions """
def run_classification_model(
    df_input,
    master_df_input,
    categories_column,
    main_text_column,
    id_column_df,
    id_column_categories,
    model_path,
    stopwords,
    accents,
    confidence_ratio = 0.8,
    match_score = 2,
    include_matched_words = 0
):
    
    # if no stopwords or accents lists were provided
    if stopwords == 0:
        stopwords = []
    elif stopwords == 1:
        stopwords = stopwords_list
    else:
        stopwords
    
    if accents == 0:
        accents = []
    elif accents == 1:
        accents = accents_list
    else:
        accents

    # spark = SparkSession.builder.appName("NLP app").getOrCreate()
    # spark = spark_session

    # df = spark.createDataFrame(df_input)
    # master_categories = spark.createDataFrame(master_df_input)

    df = df_input
    master_categories = master_df_input

    # COLS_RENAME_TARGET_TYPES = {
    #     ('model_sources', 'model_source_specialities'),
    #     ('matched_words', 'matched_speciality')
    # }

    # Clean master_degrees
    master_categories = fx.clean_text_column(
        master_categories,
        categories_column,
        stopwords,
        accents,
        "[^a-z0-9\\s]"
    )

    # Clean input
    df = fx.clean_text_column(
        df,
        main_text_column,
        stopwords,
        accents,
        "[^a-z0-9\\s]"
    )

    # Prepare bags of words
    master_degrees_words = fx.master_column_prepare(
        master_categories,
        categories_column,
        stopwords,
        accents
    )

    target_specialities = [row[0] for row in master_degrees_words.dropDuplicates(['words']).select('words').collect()]

    # Degree name classification
    df = fx.apply_multiple_spelling_corrections(
        df,
        main_text_column,
        target_specialities,
        stopwords,
        accents,
        model_path,
        confidence_ratio,
        "|"
    )

    df = df.drop("original_tokens_matched").withColumnRenamed('name_corrected',f"{main_text_column}_corrected").withColumnRenamed('matched_words','corrected_words')

    # Cast ID
    df = df.withColumn(
        "id",
        F.col(id_column_df).cast(StringType())
    )

    master_categories = master_categories.withColumn(
        "id",
        F.col(id_column_categories).cast(StringType())
    )

    # NLP Classification
    df = fx.degree_clasification(
        df,
        master_categories,
        match_score,
        f"{main_text_column}_corrected",
        categories_column,
        id_column_df,
        id_column_categories,
        include_matched_words
    )

    return df
