import gensim
import pandas as pd
import Levenshtein
import numpy as np

from pyspark.sql import Window
import re

from pyspark.sql.types import StringType, IntegerType, ArrayType
from pyspark.broadcast import Broadcast

from pyspark.sql import SparkSession
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

def train_word2vec_ml(
    df,
    text_column,
    model_path="trained_model.model",
    test_word = "industrial",
    window=5,
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
    print(model.wv.most_similar(test_word))


""" spark functions """
def run_classification_model(
    df,
    master_categories,
    categories_column,
    main_text_column,
    id_column_df,
    id_column_categories,
    model_path,
    stopwords = stopwords_list,
    accents = accents_list,
    ratio_speciality = 0.75,
    match_score = 2
):
    spark = SparkSession.builder.appName("NLP app").getOrCreate()

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
        ratio_speciality,
        "|"
    )

    df = df.drop(main_text_column, "original_tokens_matched").withColumnRenamed('name_corrected', main_text_column
    ).withColumnRenamed('model_sources', 'model_source_specialities').withColumnRenamed('matched_words', 'matched_speciality')

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
        main_text_column,
        categories_column,
        id_column_df,
        id_column_categories
    )

    # Optional: format arrays as strings
    df = df.withColumn("df_tokens", F.concat(F.lit("['"), F.concat_ws("','", F.col("df_tokens")), F.lit("']")))
    df = df.withColumn("categories_tokens", F.concat(F.lit("['"), F.concat_ws("','", F.col("master_tokens")), F.lit("']")))

    return df