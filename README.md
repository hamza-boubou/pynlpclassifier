# H1 - PyNLPclassifier

PyNLPclassifier is a python module that combines 2 features, correct the spelling of words, and classifies the data into categories
The module runs on Spark engine, uses a combination of the ML models word2vec and levenshtein distance to correct spelling against a bag of clean words, and a matching method by giving each line the equivalent category from a df through picking the catgory which has the highest amount of matching words against each row that complies with the match score

## H2 - How it works?
[![Video Title](https://i.imgur.com/MebRz7J.png)](https://youtu.be/531DrEGhXRI?si=HqTiuY2btbSBOTx2)

## H2 - Example of usage in the article:
[Example usage](https://medium.com/@hamzahbdeveloper/introducing-pynlpclassifier-correct-the-spelling-and-categorize-large-free-text-input-data-using-da1a93408f30)

**Before (raw free text input data):**
```
+---+-------------------------------------------------------------------------+
| ID|                                                                     text|
+---+-------------------------------------------------------------------------+
|  1|                                    profesr mathematics physics 1st grade|
|  2|                                            softwre developer 5 years exp|
|  3|experienced business manageemnts specoialist public relationships manager|
|  4|                                        big dta enginer spark hadoop hive|
|  5|                          chemicl enginer long experience pharma industry|
|  6|                                         softwae development professional|
|  7|                                  dermatologit medical science specialist|
+---+-------------------------------------------------------------------------+
```

**After (spelling corrected, categorized):**
```
+---+-------------------------------------------------------------------------+-------------------------------------------------------------------------+-----------------+
| ID|                                                                     text|                                                           text_corrected|Category_category|
+---+-------------------------------------------------------------------------+-------------------------------------------------------------------------+-----------------+
|  1|                                    profesr mathematics physics 1st grade|                                    profesr mathematics physics 1st grade|             NULL|
|  2|                                            softwre developer 5 years exp|                                           software developer 5 years exp|             NULL|
|  3|experienced business manageemnts specoialist public relationships manager|experienced business manageemnts specoialist public relationships manager|             NULL|
|  4|                                        big dta enginer spark hadoop hive|                                       big dta engineer spark hadoop hive|big data engineer|
|  5|                          chemicl enginer long experience pharma industry|                        chemical engineer long experience pharma industry|chemical engineer|
|  6|                                         softwae development professional|                                        software development professional|             NULL|
|  7|                                  dermatologit medical science specialist|                                 dermatologist medical science specialist|             NULL|
+---+-------------------------------------------------------------------------+-------------------------------------------------------------------------+-----------------+
```


## H2 - Functions:
```
train_ml_model(
    df,
    text_column,
    model_path="trained_model.model",
    test_word = "industrial",
    window=5,
    min_count=2,
    workers=4
)

run_classification_model(
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
)
```

## H2 - Parameters:
train_ml_model(df, "col1", "trained.model", "mechanic", 7, 2, 6)
```
*df: pandas dataframe*
*text_column the targetted column which contains free text input records*
*model_path: the output path with the .model file name where you want to save the trained word2vec model*
*test_word: a word to test the accuracy of your model*
*window: amount of words to analyze*
*min_count: minimum amount of words to consider*
*workers: nodes you want to use to train the model*
```

run_classification_model(df_input, master_df_input, "category_name", "Text", "id" , "id", "trained.model", 0, 0, 0.8 , 2)
```
*df: spark dataframe*
*master_df_input: categories dataframe with master data*
*categories_column: categories column which want to add from your master data*
*main_text_column: the targetted free text input column in which you want to process*
*id_column_df: required an auto generated index column from the spark dataframe*
*id_column_categories: auto generated index column from the master data dataframe*
*model_path: the path where your word2vec trained model sits generated from the previous function*
*stopwords: list of stopwords you want to ignore from processing*
*accents: need to replace the accents with their original form, list of accents with their replacements*
*ratio_speciality: the similarity ratio to consider in order to correct the spelling*
*match_score: minimum amount of matching words of master data to consider the candidate and place the category to the corresponding record*
*include_matched_words: if you need to include the columns of matched wordds type 1, if not type 0*
```

### H3 - stopwords and accents parameters explanation:

Stopwords: if you do not want to exclude stopwords (the, and, in, on…) from the equations you should put 0, if you want to use and exclude the default list offered by the library put 1, if you want to use your own list type variable name
Example stopwords list: ```stopwords_list = ['the', 'of', 'on', 'in', ''with', 'that', 'and', 'for']```

Accents: if you do not want to exclude accents (á, ó, é…) from the equations you should put 0, if you want to use and exclude the default list offered by the library put 1, if you want to use your own list type variable name
Example accents list: 
```
accents_list = [
    ("á", "a"),
    ("é", "e"),
    ("í", "i")
]
```

## H2 - Model metrics:
Throughout a period of testing the model, over a large amount of data, over a portion of 200k rows sample, a list of 600 categories as master data,the model corrected the spelling of 90% of the words, amd categorize 70% of the rows into categories from the master data.


## H2 - Example code application:
```
import pandas as pd
from pyspark.sql import SparkSession
import pynlpclassifier as nlp

# Creating the df
data = [
    (1, "profesr in mathematics and physics of 1st grade"),
    (2, "softwre developer with 5 years of exp"),
    (3, "experienced business manageemnts specoialist and public relationships manager"),
    (4, "big dta enginer, spark, hadoop, hive"),
    (5, "chemicl enginer with long experience in pharma industry"),
    (6, "softwae development professional"),
    (7, "dermatologit and medical science specialist"),
]

column_names = ["ID", "text"]

df_pandas = pd.DataFrame(data, columns=column_names)

# Creating the master categories data df
data = [
    (1, "Software engineer"),
    (2, "Big Data Engineer"),
    (3, "Dermatologist"),
    (4, "Chemical Engineer")
]

column_names = ["ID", "Category"]

master_data_pandas = pd.DataFrame(data, columns=column_names)

# run the training of the ML model
nlp.train_ml_model(
    df_pandas,
    "text",
    "/content/trained.model",
    "software",
    1,
    0,
    4
)

# start the spark session
spark = SparkSession.builder.appName("NLps").getOrCreate()

# convert the pandas dfs to pyspark dfs in order to use them in the next function
df = spark.createDataFrame(df_pandas)
master = spark.createDataFrame(master_data_pandas)

# run the main process
df = nlp.run_classification_model(
    df,
    master,
    "Category",
    "text",
    "ID",
    "ID",
    "/content/trained.model",
    0,
    1,
    0.8 ,
    1,
    1
)

# show portion og the output
df.show(30, truncate=50)
```


