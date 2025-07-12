NLP classifcation, spell checking and ctaegorization module

PyNLPclassifier uses a combination of word2vec ML model and Levenshtein distance to correct the spelling of the text column using a vocabolary of bag of clean words list then categorize the data

Pipelines process:

1. CLean the targeted text column
2. Train the model
3. Use the trained gensim model with a combination of double layers of levenshtein to correct the spelling of the words similar to the bag of words
4. Corssjoins the df with the master categories, tokenizes the records, tries to match the words of both sides, counts of the matching words, rank them, and picks the candidate with the highest match score

Instructions:

First thing first is to trin the word2vec model by running the function train_word2vec_ml()
Then run the function run_classification_model() which is the main engine of the model

Functions:

train_word2vec_ml(
    df,
    text_column,
    model_path="trained_model.model",
    test_word = "industrial",
    window=config.WINDOW_SIZE,
    min_count=config.MIN_COUNT,
    workers=4
)

df: pandas dataframe
text_column the targetted column which contains free text input records
model_path: the output path with the .model file name where you want to save the trained word2vec model
test_word: a word to test the accuracy of your model
window: amount of words to analyze
min_count: minimum amount of words to consider
workers: nodes you want to use to train the model

Example use: train_word2vec_ml( df, "column1", "models/trained.model", "mechanic", 7, 2, 6)

run_classification_model(
    df,
    master_categories,
    categories_column,
    main_text_column,
    id_column_df,
    id_column_categories,
    model_path,
    stopwords = stopwords_list,
    accents = accents_list,
    ratio_speciality = config.ratio_speciality,
    match_score = config.match_score
)

df: spark dataframe
master_categories: categories column which want to add from your master data
main_text_column: the targetted free text input column in which you want to process
id_column_df: required an auto generated index column from the spark dataframe
id_column_categories: auto generated index column from the master data dataframe
model_path: the path where your word2vec trained model sits generated from the previous function
stopwords: list of stopwords you want to ignore from processing
accents: need to replace the accents with their original form, list of accents with their replacements
ratio_speciality: the similarity ratio to consider in order to correct the spelling
match_score: minimum amount of matching words of master data to consider the candidate and place the category to the corresponding record

Example use: run_classification_model(df, master_df, "category", "column1", "id", "id", "models/trained.model", stopwords_list, accents_list, 0.80, 2)

Made with love by Hamza B.
