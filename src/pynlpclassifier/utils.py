import gensim
import pandas as pd
import Levenshtein
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window
from functools import reduce
import re

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, IntegerType, StructType, StructField, ArrayType, FloatType
from pyspark.broadcast import Broadcast
import glob

stopwords = [
    'de', 'la', 'el', 'y', 'en', 'con', 'del', 'los', 'las', 'para', 'por', 'a',
    'un', 'una', 'al', 'o', 'u', 'e', 'se', 'su', 'que', 'como',
    'the', 'of', 'on', 'in', 'by', 'with', 'that', 'and', 'for'
]

special_letters = [
    ("á", "a"),
    ("é", "e"),
    ("í", "i"),
    ("ó", "o"),
    ("ú", "u"),
    ("ñ", "n")
]

""" Clean text in pandas """
def clean_text(text):
    """
    Removes stop words, special characters, and replaces letters with accents
    """

    # Convert to lowercase
    text = text.lower()

    # Replace special characters
    for original, replacement in special_letters:
        text = text.replace(original, replacement)

    # Remove stopwords
    # Create a regex pattern to match whole words for stopwords
    # Using \b for word boundaries to avoid partial matches (e.g., 'the' matching inside 'another')
    # Sort stopwords by length in descending order to avoid issues with shorter words being replaced first
    # that are part of longer stopwords (e.g., "del" before "de")
    sorted_stopwords = sorted(stopwords, key=len, reverse=True)

    for stop_word in sorted_stopwords:
        # Using re.sub with word boundaries and ignoring case
        text = re.sub(r'\b' + re.escape(stop_word) + r'\b', '', text)

    # Remove any extra spaces left after removing stopwords
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r"\.|\,|\?|\!|\@|\#|\$|\%|\^|\&|\*|\(|\)|\-|\+|\=|\{|\}|\[|\]|\:|\;|\'|\"", ' ', text).strip()

    return text

"""" clean column spark """
def clean_text_column(
    df: DataFrame,
    column_name: str,
    stopwords: list,
    special_letters: list,
    punctuation_regex: str = "[^a-z\\s]"
) -> DataFrame:
    """
    Cleans a text column in a PySpark DataFrame by performing the following operations:
    1. Lowercasing the text.
    2. Removing specified punctuations.
    3. Replacing special characters (e.g., accented letters).
    4. Removing extra spaces (collapsing multiple spaces to one, and trimming).
    5. Removing specified stopwords.

    Args:
        df (DataFrame): The input PySpark DataFrame.
        column_name (str): The name of the text column to clean.
        stopwords (list): A list of strings representing stopwords to remove.
        special_letters (list): A list of tuples, where each tuple is (old_char, new_char)
                                 for character replacement.
        punctuation_regex (str, optional): A regular expression string to identify
                                           characters to be removed as punctuation.
                                           Defaults to "[^a-z\\s]" (removes anything
                                           that is not a lowercase letter or a space).

    Returns:
        DataFrame: A new DataFrame with the cleaned text in a column named
                   'cleaned_{column_name}'.
    """

    # Start with the original column and create a working column
    cleaned_df = df.withColumn(f"cleaned_{column_name}", F.col(column_name))

    # 1. Lowercase the column
    cleaned_df = cleaned_df.withColumn(
        f"cleaned_{column_name}",
        F.lower(F.col(f"cleaned_{column_name}"))
    )

    # 2. Remove all punctuations
    cleaned_df = cleaned_df.withColumn(
        f"cleaned_{column_name}",
        F.regexp_replace(F.col(f"cleaned_{column_name}"), punctuation_regex, "")
    )

    # 3. Replace special letters
    for old_char, new_char in special_letters:
        # Using re.escape for old_char to ensure special regex characters are treated literally
        cleaned_df = cleaned_df.withColumn(
            f"cleaned_{column_name}",
            F.regexp_replace(F.col(f"cleaned_{column_name}"), re.escape(old_char), new_char)
        )

    # 4. Remove extra spaces (normalize to single spaces and trim)
    cleaned_df = cleaned_df.withColumn(
        f"cleaned_{column_name}",
        F.regexp_replace(F.col(f"cleaned_{column_name}"), "\\s+", " ") # Replace multiple spaces with single space
    )
    cleaned_df = cleaned_df.withColumn(
        f"cleaned_{column_name}",
        F.trim(F.col(f"cleaned_{column_name}")) # Remove leading/trailing spaces
    )

    # 5. Remove stopwords
    # First, split the cleaned string into an array of words
    words_array_col_name = f"words_array_{column_name}"
    cleaned_df = cleaned_df.withColumn(
        words_array_col_name,
        F.split(F.col(f"cleaned_{column_name}"), " ")
    )

    # Filter out stopwords using array_except
    filtered_words_array_col_name = f"filtered_words_array_{column_name}"
    cleaned_df = cleaned_df.withColumn(
        filtered_words_array_col_name,
        F.array_except(F.col(words_array_col_name), F.lit(stopwords))
    )

    # Join the words back into a string, separated by space
    cleaned_df = cleaned_df.withColumn(
        column_name, # Overwrite the working column with the final cleaned string
        F.concat_ws(" ", F.col(filtered_words_array_col_name))
    )

    # Drop intermediate array columns if not needed in the final output
    cleaned_df = cleaned_df.drop(words_array_col_name, filtered_words_array_col_name, f"cleaned_{column_name}")

    return cleaned_df

def master_column_prepare(
    master_categories,
    categories_column,
    stopwords,
    accents
):
    
    master_categories = clean_text_column(
        master_categories,
        categories_column,
        stopwords,
        accents,
        "[^a-z0-9\\s]"
    )

    master_categories = master_categories.withColumn(
        'words',
        F.split(F.col(categories_column), " ")
    )

    master_categories = master_categories.withColumn(
        'words',
        F.explode(F.col("words"))
    )

    return master_categories


""" test correcy mutiple words """
udf_return_schema_multiple_matches = StructType([
    StructField("name_corrected", StringType(), True),
    StructField("matched_words", StringType(), True),
    StructField("original_tokens_matched", StringType(), True),
    StructField("model_sources", StringType(), True) # Store as string to join ints
])

def apply_multiple_spelling_corrections(
    spark_df,
    name_column, # The name of the column containing text to correct
    target_words=None,
    stopwords=None,
    special_letters=None,
    model_path=None,
    threshold=0.7,
    delimiter="," # Delimiter for joining multiple matched words/tokens
):
    """
    Applies word matching and correction logic to a PySpark DataFrame,
    correcting multiple words per row and adding detailed match information.

    Args:
        spark_df (pyspark.sql.DataFrame): The input Spark DataFrame.
        name_column (str): The name of the column to perform spell correction on.
        target_words (list): List of words to match against.
        stopwords (list): List of stopwords to filter during preprocessing.
        special_letters (list): List of (original, replacement) tuples for special chars.
        model_path (str): Path to the trained Word2Vec model.
        threshold (float): Minimum similarity score to consider a word a match.
        delimiter (str): The character to use for separating multiple words in output columns.

    Returns:
        pyspark.sql.DataFrame: The DataFrame with 'name_corrected', 'matched_words',
                               'original_tokens_matched', and 'model_sources' columns.
    """
    if target_words is None or stopwords is None or special_letters is None or model_path is None:
        raise ValueError("target_words, stopwords, special_letters error")

    # --- Define the actual UDF logic (nested within this function) ---
    @F.udf(udf_return_schema_multiple_matches)
    def _find_multiple_similar_word_details_udf(text):
        # Initialize lists to store all valid matches for this row
        all_original_tokens = []
        all_matched_words = []
        all_model_sources = []
        current_text_to_process = text # This string will be modified for replacements

        if text is None:
            return None, None, None, None

        # Load the Word2Vec model inside the UDF using a global cache.
        if 'cached_model' not in globals():
            try:
                globals()['cached_model'] = gensim.models.Word2Vec.load(model_path)
            except FileNotFoundError:
                print(f"Warning: Word2Vec model not found at {model_path} on executor. Word2Vec matching disabled.")
                globals()['cached_model'] = None
            except Exception as e:
                print(f"Error loading Word2Vec model on executor: {e}. Word2Vec matching disabled.")
                globals()['cached_model'] = None
        local_model = globals()['cached_model']

        # Preprocessing helper function
        def _preprocess_text(input_text_for_preprocess):
            input_text_for_preprocess = input_text_for_preprocess.lower()
            for original, replacement in special_letters:
                input_text_for_preprocess = input_text_for_preprocess.replace(original, replacement)
            tokens = gensim.utils.simple_preprocess(input_text_for_preprocess)
            return tokens

        processed_tokens_in_text = _preprocess_text(text)

        # Dictionary to store the best match found for each unique token in the input text
        # Key: original token from text, Value: (matched_word, model_source, similarity_score)
        best_matches_for_each_token = {}

        # --- Phase 1 & 2: Collect all best matches per input token ---
        for word_in_text in processed_tokens_in_text:
            # Skip stopwords for matching logic, but keep for replacement
            if word_in_text in stopwords:
                continue

            current_best_match = None
            current_best_similarity = -1
            current_model_source = None

            # Word2Vec check
            if local_model and word_in_text in local_model.wv.key_to_index:
                for target_word in target_words:
                    if target_word in local_model.wv.key_to_index:
                        similarity = local_model.wv.similarity(word_in_text, target_word)
                        if similarity > current_best_similarity and similarity >= threshold:
                            current_best_similarity = similarity
                            current_best_match = target_word
                            current_model_source = 1 # Word2Vec

            # Levenshtein fallback (if no Word2Vec match or for target_words not in vocab)
            # We run Levenshtein for all, but prioritize Word2Vec if its similarity is higher
            for target_word in target_words:
                lev_distance = Levenshtein.distance(word_in_text, target_word)
                max_len = max(len(word_in_text), len(target_word))
                if max_len > 0:
                    lev_similarity = 1 - (lev_distance / max_len)
                    # If Levenshtein is better or current_best_match is still None
                    if lev_similarity > current_best_similarity and lev_similarity >= threshold:
                        current_best_similarity = lev_similarity
                        current_best_match = target_word
                        current_model_source = 3 # Levenshtein (could be 2 or 3, simplifying to 3 for any Levenshtein catch for now)
                        # If a Word2Vec match was previously found, but Levenshtein is also good
                        # and better, update the source code. If Word2Vec was already best, keep it.
                        if local_model and word_in_text in local_model.wv.key_to_index and target_word in local_model.wv.key_to_index:
                            if local_model.wv.similarity(word_in_text, target_word) >= current_best_similarity:
                                current_model_source = 1 # Prefer Word2Vec if similar score

            if current_best_match:
                # Store the best match found for this specific word_in_text
                best_matches_for_each_token[word_in_text] = (current_best_match, current_model_source)

        # --- Perform all replacements and collect results ---
        # Sort matches by the length of the original token in descending order
        # to prevent shorter matches from interfering with longer ones
        sorted_matches_keys = sorted(best_matches_for_each_token.keys(), key=len, reverse=True)

        for original_token in sorted_matches_keys:
            matched_word, model_source = best_matches_for_each_token[original_token]

            # Append to lists for output
            all_original_tokens.append(original_token)
            all_matched_words.append(matched_word)
            all_model_sources.append(str(model_source))

            # Perform replacement on the current_text_to_process
            # Using regex with word boundaries and case-insensitivity
            pattern = r'(?i)\b' + re.escape(original_token) + r'\b'
            current_text_to_process = re.sub(pattern, matched_word, current_text_to_process, count=1)


        # Join lists by delimiter for output columns
        joined_matched_words = delimiter.join(all_matched_words) if all_matched_words else None
        joined_original_tokens = delimiter.join(all_original_tokens) if all_original_tokens else None
        joined_model_sources = delimiter.join(all_model_sources) if all_model_sources else None

        # Return a tuple matching the udf_return_schema_multiple_matches
        return current_text_to_process, joined_matched_words, joined_original_tokens, joined_model_sources

    # --- Apply the UDF to the DataFrame and extract results ---
    # The UDF will add a temporary struct column 'udf_output'
    df_with_udf_results = spark_df.withColumn(
        "udf_output", _find_multiple_similar_word_details_udf(F.col(name_column))
    )

    # Extract the individual fields from the 'udf_output' struct into new columns
    df_final = df_with_udf_results.withColumn("name_corrected", F.col("udf_output.name_corrected")) \
                                 .withColumn("matched_words", F.col("udf_output.matched_words")) \
                                 .withColumn("original_tokens_matched", F.col("udf_output.original_tokens_matched")) \
                                 .withColumn("model_sources", F.col("udf_output.model_sources")) \
                                 .drop("udf_output") # Drop the intermediate struct column

    return df_final


""" add degree_name column with best match """


""" new method """
def degree_clasification(
    df,
    master_degrees,
    score,
    main_text_column,
    categories_column,
    id_column_df,
    id_column_categories
):
    """
    Classifies degrees by intersecting tokens from matched_speciality and master_degree_name.

    Args:
        df (DataFrame): The main DataFrame containing 'id', 'name', 'matched_speciality'
                        (pipe-separated string), and other relevant columns.
        master_degrees (DataFrame): The DataFrame containing 'degree_name' and other
                                    master degree information.
        score (float): A threshold score (though not directly used in the provided logic for filtering).

    Returns:
        DataFrame: A DataFrame with the best matching master degree for each entry in df,
                   based on token intersection score.
    """

    # 1. Prepare matched_speciality column: replace '|' with space and convert to lowercase
    # This prepares the string for tokenization.
    df_prepared = df.withColumn(
        "matched_speciality_cleaned",
        F.lower(F.regexp_replace(F.col("matched_speciality"), "\\|", " "))
    )

    # 2. Tokenizing degree_name from master_degrees and matched_speciality from df
    # Split the cleaned strings into arrays of words (tokens).
    df_with_tokens = df_prepared.withColumn(
        "df_tokens",
        F.split(F.col("matched_speciality_cleaned"), " ")
    )

    master_degrees_with_tokens = master_degrees.withColumn(
        "master_tokens",
        F.split(F.lower(F.col(categories_column)), " ") # Assuming 'degree_name' is the column for master degree names
    ).withColumnRenamed(id_column_categories,'master_category_id')

    # 3. Cross join the tables
    # This creates all possible pairs between records in df and master_degrees.
    df_cross_joined = df_with_tokens.crossJoin(master_degrees_with_tokens)

    # 4. Calculate match score by intersecting the token arrays
    # The match score is the count of common tokens between the education and master degree descriptions.
    df_score = df_cross_joined.withColumn(
        "match_score",
        F.size(
            F.array_intersect(
                F.col("df_tokens"), F.col("master_tokens")
            )
        )
    )

    # 5. Rank matches for each id
    # This assigns a rank to each master degree match for a given 'id' based on 'match_score'.
    window_spec = Window.partitionBy(id_column_df).orderBy(F.col("match_score").desc())

    df_rank = df_score.withColumn(
        "rank",
        F.row_number().over(window_spec)
    )

    # 6. Pick rows with rank 1 (the best match)
    best_matches_df = df_rank.filter(
        (F.col('rank') == 1)
        | (F.col('rank').isNull())
    )

    best_matches_df = best_matches_df.filter(F.col('match_score') > score)

    return best_matches_df