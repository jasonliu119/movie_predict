# from __future__ import print_function
# encoding=utf8  
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
import json
import operator
from collections import Counter

tf.logging.set_verbosity(tf.logging.ERROR)

movie_dataframe = pd.read_csv("train.csv")

def clean_data(train):
    train.loc[train['id'] == 16,'revenue'] = 192864          # Skinning
    train.loc[train['id'] == 90,'budget'] = 30000000         # Sommersby          
    train.loc[train['id'] == 118,'budget'] = 60000000        # Wild Hogs
    train.loc[train['id'] == 149,'budget'] = 18000000        # Beethoven
    train.loc[train['id'] == 313,'revenue'] = 12000000       # The Cookout 
    train.loc[train['id'] == 451,'revenue'] = 12000000       # Chasing Liberty
    train.loc[train['id'] == 464,'budget'] = 20000000        # Parenthood
    train.loc[train['id'] == 470,'budget'] = 13000000        # The Karate Kid, Part II
    train.loc[train['id'] == 513,'budget'] = 930000          # From Prada to Nada
    train.loc[train['id'] == 797,'budget'] = 8000000         # Welcome to Dongmakgol
    train.loc[train['id'] == 819,'budget'] = 90000000        # Alvin and the Chipmunks: The Road Chip
    train.loc[train['id'] == 850,'budget'] = 90000000        # Modern Times
    train.loc[train['id'] == 1007,'budget'] = 2              # Zyzzyx Road 
    train.loc[train['id'] == 1112,'budget'] = 7500000        # An Officer and a Gentleman
    train.loc[train['id'] == 1131,'budget'] = 4300000        # Smokey and the Bandit   
    train.loc[train['id'] == 1359,'budget'] = 10000000       # Stir Crazy 
    train.loc[train['id'] == 1542,'budget'] = 1              # All at Once
    train.loc[train['id'] == 1570,'budget'] = 15800000       # Crocodile Dundee II
    train.loc[train['id'] == 1571,'budget'] = 4000000        # Lady and the Tramp
    train.loc[train['id'] == 1714,'budget'] = 46000000       # The Recruit
    train.loc[train['id'] == 1721,'budget'] = 17500000       # Cocoon
    train.loc[train['id'] == 1865,'revenue'] = 25000000      # Scooby-Doo 2: Monsters Unleashed
    train.loc[train['id'] == 1885,'budget'] = 12             # In the Cut
    train.loc[train['id'] == 2091,'budget'] = 10             # Deadfall
    train.loc[train['id'] == 2268,'budget'] = 17500000       # Madea Goes to Jail budget
    train.loc[train['id'] == 2491,'budget'] = 6              # Never Talk to Strangers
    train.loc[train['id'] == 2602,'budget'] = 31000000       # Mr. Holland's Opus
    train.loc[train['id'] == 2612,'budget'] = 15000000       # Field of Dreams
    train.loc[train['id'] == 2696,'budget'] = 10000000       # Nurse 3-D
    train.loc[train['id'] == 2801,'budget'] = 10000000       # Fracture
    train.loc[train['id'] == 335,'budget'] = 2 
    train.loc[train['id'] == 348,'budget'] = 12
    train.loc[train['id'] == 470,'budget'] = 13000000 
    train.loc[train['id'] == 513,'budget'] = 1100000
    train.loc[train['id'] == 640,'budget'] = 6 
    train.loc[train['id'] == 696,'budget'] = 1
    train.loc[train['id'] == 797,'budget'] = 8000000 
    train.loc[train['id'] == 850,'budget'] = 1500000
    train.loc[train['id'] == 1199,'budget'] = 5 
    train.loc[train['id'] == 1282,'budget'] = 9               # Death at a Funeral
    train.loc[train['id'] == 1347,'budget'] = 1
    train.loc[train['id'] == 1755,'budget'] = 2
    train.loc[train['id'] == 1801,'budget'] = 5
    train.loc[train['id'] == 1918,'budget'] = 592 
    train.loc[train['id'] == 2033,'budget'] = 4
    train.loc[train['id'] == 2118,'budget'] = 344 
    train.loc[train['id'] == 2252,'budget'] = 130
    train.loc[train['id'] == 2256,'budget'] = 1 
    train.loc[train['id'] == 2696,'budget'] = 10000000
    return train

movie_dataframe = clean_data(movie_dataframe)
movie_dataframe = movie_dataframe.reindex(np.random.permutation(movie_dataframe.index))
movie_dataframe.reset_index(inplace=True, drop=True)

# budget: + 1 and then log norm
# genres: one-hot encoding, categorical_column_with_vocabulary_list
# homepage: has or not
# 

# def describe_lang():
#     value_dict = {}
#     for l in movie_dataframe['original_language']:
#         try:
#             if l in value_dict:
#                 value_dict[l] += 1
#             else:
#                 value_dict[l] = 1
#         except:
#             pass
#     print(value_dict.keys())
#     print(len(value_dict))
#     print(len(np.array(movie_dataframe['original_language'])))

import ast
dict_columns = ['belongs_to_collection', 'genres', 'production_companies',
                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']

def text_to_dict(df):
    for column in dict_columns:
        df[column] = df[column].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x) )
    return df

# print("Converting text in original dataframe into dict with ast.literal_eval...\n\n"
# to_dict_movie_dataframe = movie_dataframe.copy()
# to_dict_movie_dataframe = text_to_dict(to_dict_movie_dataframe)

# use lambda to get the top list of a certain feature after converting the text to dict
def get_top(feature_name, length = 30, field = 'name'):
    x = to_dict_movie_dataframe[feature_name].apply(lambda x : [x[i][field] for i in range(len(x))] if x != {} else []).values
    top = [x[0] for x in Counter([i for j in x for i in j]).most_common(length)]
    print(" -- Top " + feature_name + "\n" + str(top))
    print('\n\n')
    return top

# def describe_json_feature(feature_name, field = 'name'):
#     gen_dict = {}
#     null_count = 0
#     for s in movie_dataframe[feature_name]:
#         print(s)
#         try:
#             j = json.loads(str(s).replace('\'', '\"'))

#             for gen in j:
#                 type_name = gen[field]
#                 if type_name in gen_dict:
#                     gen_dict[type_name] += 1
#                 else:
#                     gen_dict[type_name] = 1
#         except:
#             null_count += 1
#     sorted_dict = dict(sorted(gen_dict.items(), key=lambda x : x[1], reverse=True)[:20])
#     print(sorted_dict)
#     print(sorted_dict.keys())
#     print("nan count " + str(null_count))

def preprocess_targets(df):
  output_targets = pd.DataFrame()
  output_targets["revenue"] = np.log1p(df["revenue"])
  return output_targets

def log_linear_process(raw_series):
    # log and normalize
    series = raw_series.apply(lambda x:math.log(x + 1))
    min_val = series.min()
    max_val = series.max()
    scale = (max_val - min_val) / 1.0
    return series.apply(lambda x: (x - min_val) / scale)

def linear_process(series):
    # normalize
    min_val = series.min()
    max_val = series.max()
    scale = (max_val - min_val) / 1.0
    return series.apply(lambda x: (x - min_val) / scale)

def process_genres(raw_series):
    ret = raw_series.copy()
    for i in range(raw_series.size):
        one_movie = ''
        try:
            # e.g., '[{"id": 28, "name": "Action"}, {"id": 35, "name": "Comedy"}, {"id": 18, "name": "Drama"}]'
            j = json.loads(raw_series[i].replace('\'', '\"'))

            for gen in j:
                one_movie += ' '
                one_movie += gen['name']
        except:
            continue
        ret[i] = one_movie
    return ret.str.split()

from datetime import datetime

def date(x, type):
    x=str(x)
    year=x.split('/')[2]
    if int(year)<19:
        date = x[:-2]+'20'+year
    else:
        date = x[:-2]+'19'+year
    date = datetime.strptime(date,'%m/%d/%Y')

    if type == 'weekday':
        return date.weekday()
    if type == 'month':
        return date.month
    if type == 'year':
        return date.year

    return date.day

genres_v_list = [u'Mystery', u'Romance', u'Family', u'Science Fiction', u'Horror',
        u'Crime', u'Drama', u'Fantasy', u'Western', u'Animation', u'Music',
        u'Adventure', u'Foreign', u'Action', u'TV Movie', u'Comedy', u'Documentary',
        u'War', u'Thriller', u'History']

# get_top('production_companies')
company_v_list = ['Warner Bros.', 'Universal Pictures', 'Paramount Pictures', 'Twentieth Century Fox Film Corporation',
    'Columbia Pictures', 'Metro-Goldwyn-Mayer (MGM)', 'New Line Cinema', 'Touchstone Pictures', 'Walt Disney Pictures',
    'Columbia Pictures Corporation', 'TriStar Pictures', 'Relativity Media', 'Canal+', 'United Artists', 'Miramax Films',
    'Village Roadshow Pictures', 'Regency Enterprises', 'BBC Films', 'Dune Entertainment', 'Working Title Films',
    'Fox Searchlight Pictures', 'Lionsgate', 'StudioCanal', 'DreamWorks SKG', 'Fox 2000 Pictures', 'Hollywood Pictures',
    'Orion Pictures', 'Summit Entertainment', 'Dimension Films', 'Amblin Entertainment']

# get_top('spoken_languages', 15, 'iso_639_1')
language_v_list = ['en', 'fr', 'es', 'de', 'ru', 'it', 'ja', 'zh', 'hi', 'pt', 'ar', 'ko', 'cn', 'ta', 'pl']

# get_top('Keywords', 30)
keyword_v_list = ['woman director', 'independent film', 'duringcreditsstinger', 'murder', 'based on novel',
    'violence', 'sport', 'biography', 'aftercreditsstinger', 'revenge', 'dystopia', 'friendship', 'sex', 'suspense',
    'sequel', 'love', 'police', 'teenager', 'nudity', 'female nudity', 'drug', 'prison', 'musical', 'high school',
    'los angeles', 'new york', 'family', 'father son relationship', 'investigation', 'kidnapping']

# get_top('cast', 30)
cast_v_list = ['Robert De Niro', 'Samuel L. Jackson', 'Morgan Freeman', 'Bruce Willis', 'Susan Sarandon',
    'J.K. Simmons', 'Liam Neeson', 'John Turturro', 'Bruce McGill', 'Willem Dafoe', 'Forest Whitaker',
    'Nicolas Cage', 'Bill Murray', 'Owen Wilson', 'Jason Statham', 'John Goodman', 'Sigourney Weaver',
    'Mel Gibson', 'Sylvester Stallone', 'Keith David', 'Dennis Quaid', 'Robert Duvall', 'Michael Caine',
    'Matt Damon', 'Ed Harris', 'Denzel Washington', 'Frank Welker', 'George Clooney', 'Richard Jenkins', 'Christopher Walken']

# get_top('crew', 30)
crew_v_list = ['Avy Kaufman', 'Robert Rodriguez', 'Deborah Aquila', 'James Newton Howard', 'Mary Vernieu',
    'Jerry Goldsmith', 'Luc Besson', 'Steven Spielberg', 'Francine Maisler', 'Tricia Wood', 'James Horner',
    'Kerry Barden', 'Janet Hirshenson', 'Harvey Weinstein', 'Bob Weinstein', 'Jane Jenkins', 'John Debney',
    'John Papsidera', 'Francis Ford Coppola', 'Hans Zimmer', 'Billy Hopkins', 'Danny Elfman', 'Mindy Marin',
    'Hans Bjerno', 'Alan Silvestri', 'Tim Bevan', 'Mark Isham', 'Neal H. Moritz', 'Sarah Finn', 'Arnon Milchan']

# get_top('belongs_to_collection', 30)
collection_v_list = ['James Bond Collection', 'Friday the 13th Collection', 'The Pink Panther (Original) Collection',
    'Pok\xc3\xa9mon Collection', 'Police Academy Collection', 'Transformers Collection', 'Rocky Collection', 'Rambo Collection',
    "Child's Play Collection", 'Alien Collection', 'Paranormal Activity Collection', 'Ice Age Collection', 'Resident Evil Collection',
    'The Fast and the Furious Collection', 'Indiana Jones Collection', 'Rush Hour Collection', 'The Dark Knight Collection',
    'Scary Movie Collection', 'Qatsi Collection', 'Missing in Action Collection', 'Three Heroes Collection', '[REC] Collection',
    'Pirates of the Caribbean Collection', 'The Jaws Collection', 'Halloween Collection', 'Alex Cross Collection', 'Mexico Trilogy',
    'Planet of the Apes Original Collection', 'Diary of a Wimpy Kid Collection', 'The Vengeance Collection']

def replace_character(s):
    return s.replace(' ', '_').replace('+', '_').replace('(', '_').replace(')', '_').replace('\'', '_').replace(
        'é', '_').replace('[', '_').replace(']', '_')

def convert_to_multi_hot_column(data_frame, feature_name, vocabulary_list=[], should_drop_original=True):
    df = data_frame.copy()
    for v in vocabulary_list:
        df[replace_character(v)] = data_frame[feature_name].apply(lambda x: 1 if (not pd.isna(x) and v in x) else 0)
    if should_drop_original:
        df.drop([feature_name], axis=1, inplace=True)
    return df

def get_release_month_year_count(pf):
    time_release_dict = {}
    time_series = pf['release_year'].astype(str) + pf['release_month'].astype(str)
    for time in time_series:
        if time in time_release_dict:
            time_release_dict[time] += 1
        else:
            time_release_dict[time] = 1
    ret = time_series.apply(lambda x: time_release_dict[x])
    return ret

def preprocess_features(dataframe):
    selected_features = dataframe[
    [
    "budget",
    "genres",
    "homepage",
    "original_language",
    "popularity",
    "release_date",
    "production_companies",
    "runtime",
    "spoken_languages",
    "Keywords",
    "cast",
    "crew",
    "belongs_to_collection",
    "overview",
    "imdb_id", # for joining with additional features
     ]]

    # pf means processed features
    pf = selected_features.copy()

    pf["budget"] = np.log1p(pf["budget"]).fillna(0)

    pf = convert_to_multi_hot_column(pf, "genres", genres_v_list)

    pf["homepage"] = (pf["homepage"].isnull() == True).astype('int').fillna(0)

    pf["popularity"] = pf["popularity"].fillna(0)

    pf["original_language"] = pf["original_language"].fillna('en')

    # pf["release_date"] = pf["release_date"].apply(lambda x : date(x))
    import random
    pf['release_day'] = pf['release_date'].apply(lambda x: date(x, 'weekday') if not pd.isna(x) else random.randint(1,7)).astype(int)
    pf['release_month'] = pf['release_date'].apply(lambda x: date(x, 'month') if not pd.isna(x) else random.randint(1,12)).astype(int)
    pf['release_year'] = pf['release_date'].apply(lambda x: date(x, 'year') if not pd.isna(x) else 1990)

    pf['_budget_year_ratio'] = dataframe['budget'] / (pf['release_year'] * pf['release_year'])
    pf['_budget_year_ratio_2'] = np.log1p(dataframe['budget'] / pf['release_year'])
    pf['_budget_runtime_ratio'] = linear_process((dataframe['budget'] / (pf['runtime'] + 1)).fillna(0))
    pf['_budget_runtime_ratio_2'] = np.log1p((dataframe['budget'] * pf['runtime']).fillna(0))

    pf['_releaseYear_popularity_ratio'] = np.log1p(pf['release_year'] / (pf['popularity'].fillna(0) + 1))
    pf['_releaseYear_popularity_ratio2'] = np.log1p(pf['popularity'] / pf['release_year'])

    # the count of the movies that was released in the same month
    # pf['release_month_year_count'] = get_release_month_year_count(pf)

    pf['release_year'] = linear_process(pf['release_year']).fillna(1990)
    pf.drop(['release_date'], axis=1, inplace=True)

    pf["popularity"] = np.log1p(pf["popularity"]).fillna(0)

    pf = convert_to_multi_hot_column(pf, "production_companies", company_v_list)

    pf['runtime'] = linear_process(pf['runtime']).fillna(0)

    pf = convert_to_multi_hot_column(pf, "spoken_languages", language_v_list)

    pf = convert_to_multi_hot_column(pf, "Keywords", keyword_v_list)

    pf = convert_to_multi_hot_column(pf, "cast", cast_v_list)

    pf = convert_to_multi_hot_column(pf, "crew", crew_v_list)

    pf = convert_to_multi_hot_column(pf, "belongs_to_collection", collection_v_list)

    pf["overview_size"] = np.log1p(pf["overview"].apply(lambda x : len(x.split(' ')) if not pd.isna(x) else 0))
    pf.drop(['overview'], axis=1, inplace=True)

    # for k in pf:
    #     print(k + " --> " + str(pf[k].size))

    # for column in pf:
    #     print('checking column ' + column
    #     for i in range(pf[column].size):
    #         if pd.isna(pf[column][i]):
    #             print(" -- isna row index " + str(i)
    # for c in pf:
    #     print(' -- column ' + c
    #     print(pf[c].describe()

    return pf

def construct_feature_columns(input_features):
    ret = set()

    ret.add(tf.feature_column.numeric_column('budget'))

    multi_hop_columns = genres_v_list + company_v_list + language_v_list + keyword_v_list + \
        cast_v_list + crew_v_list + collection_v_list
    for v in multi_hop_columns:
        ret.add(tf.feature_column.numeric_column(replace_character(v)))

    ret.add(tf.feature_column.numeric_column('homepage'))

    original_language = tf.feature_column.categorical_column_with_vocabulary_list(
        'original_language', language_v_list)
    ret.add(tf.feature_column.indicator_column(original_language))


    ret.add(tf.feature_column.numeric_column('popularity'))

    ret.add(tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_identity('release_day', 8)))
    ret.add(tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_identity('release_month', 13)))
    ret.add(tf.feature_column.numeric_column('release_year'))
    ret.add(tf.feature_column.numeric_column('_budget_year_ratio'))

    ret.add(tf.feature_column.numeric_column('runtime'))

    for feature in ['_releaseYear_popularity_ratio', '_releaseYear_popularity_ratio2', '_budget_year_ratio_2',
        '_budget_runtime_ratio', "overview_size",
        '_budget_runtime_ratio_2',
        ]:
        ret.add(tf.feature_column.numeric_column(feature))

    # ret.add(tf.feature_column.numeric_column('release_month_year_count'))

    return ret

def my_input_fn(df_features, targets, batch_size=1, shuffle=True, num_epochs=None):    
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(df_features).items()}  
    
    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(1000)
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels



def train_nn_regression_model(
    my_optimizer,
    steps,
    batch_size,
    hidden_units,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets,
    model_dir=None,
    warm_start_from=None):
  periods = 10
  steps_per_period = steps / periods
  
  # Create a DNNRegressor object.
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  dnn_regressor = tf.estimator.DNNRegressor(
      feature_columns=construct_feature_columns(training_examples),
      hidden_units=hidden_units,
      optimizer=my_optimizer,
      model_dir=model_dir,
      warm_start_from=warm_start_from
  )
  
  # Create input functions.
  training_input_fn = lambda: my_input_fn(training_examples, 
                                          training_targets["revenue"],
                                          batch_size=batch_size,
                                          shuffle=False)
  predict_training_input_fn = lambda: my_input_fn(training_examples, 
                                                  training_targets["revenue"], 
                                                  num_epochs=1, 
                                                  shuffle=False)
  predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
                                                    validation_targets["revenue"], 
                                                    num_epochs=1, 
                                                    shuffle=False)

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print("Training model...")
  print("RMSE (on training data):")
  training_rmse = []
  validation_rmse = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    dnn_regressor.train(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
    # Take a break and compute predictions.
    training_predictions = dnn_regressor.predict(input_fn=predict_training_input_fn)
    training_predictions = np.array([item['predictions'][0] for item in training_predictions])
    
    validation_predictions = dnn_regressor.predict(input_fn=predict_validation_input_fn)
    validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
    
    # Compute training and validation loss.
    training_root_mean_squared_log_error = math.sqrt(
        metrics.mean_squared_error(training_predictions, training_targets))
    validation_root_mean_squared_log_error = math.sqrt(
        metrics.mean_squared_error(validation_predictions, validation_targets))
    # Occasionally print(the current loss.
    print('  -------------------------------------------------')
    print("  period %02d train rmsle: %0.2f" % (period, training_root_mean_squared_log_error))
    print("  period %02d validation rmsle: %0.2f" % (period, validation_root_mean_squared_log_error))
    print('  -------------------------------------------------')
    # Add the loss metrics from this period to our list.
    training_rmse.append(training_root_mean_squared_log_error)
    validation_rmse.append(validation_root_mean_squared_log_error)
  print("Model training finished.")

  # Output a graph of loss metrics over periods.
  plt.ylabel("RMSE")
  plt.xlabel("Periods")
  plt.title("Root Mean Squared Error vs. Periods")
  plt.tight_layout()
  plt.plot(training_rmse, label="training")
  plt.plot(validation_rmse, label="validation")
  plt.legend()

  print("Final RMSLE (on training data):   %0.2f" % training_root_mean_squared_log_error)
  print("Final RMSLE (on validation data): %0.2f" % validation_root_mean_squared_log_error)

  return dnn_regressor, training_rmse, validation_rmse

dnn_hidden_units = [512, 256, 128]

def train_predict(training_examples, training_targets, validation_examples, validation_targets, model_dir=None, warm_start_from=None):
    return train_nn_regression_model(
        my_optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.001, l2_regularization_strength=0.005),
        steps=10000,
        batch_size=50,
        hidden_units=dnn_hidden_units,
        training_examples=training_examples,
        training_targets=training_targets,
        validation_examples=validation_examples,
        validation_targets=validation_targets,
        model_dir=model_dir,
        warm_start_from=warm_start_from)

def hold_validation_set(training_rate = 0.8):
    validation_rate = 1 - training_rate
    training_number = int(movie_dataframe.shape[0] * training_rate)
    validation_number = movie_dataframe.shape[0] - training_number

    training_examples = preprocess_features(movie_dataframe.head(training_number))
    training_targets = preprocess_targets(movie_dataframe.head(training_number))

    validation_examples = preprocess_features(movie_dataframe.tail(validation_number))
    validation_targets = preprocess_targets(movie_dataframe.tail(validation_number))

    train_predict(training_examples, training_targets, validation_examples, validation_targets)

def k_fold(k=10):
    validation_rmse_l = []
    for i in range(k):
        v_size = int(movie_dataframe.shape[0] / k) # k = 5, v_size = 3000 / 5 = 600
        v_start_row = i * v_size # 0, 600, 1200, 1800, 2400

        def f(x):
            print("------" + str(x))
            return x >= v_start_row and x < v_start_row + v_size
        v_df = movie_dataframe.iloc[v_start_row: v_start_row + v_size - 1]

        t_df = v_df
        if i == 0:
            t_df = movie_dataframe.iloc[v_start_row + v_size:]
        elif i == k - 1:
            t_df = movie_dataframe.iloc[0: v_start_row - 1]
        else:
            df_left = movie_dataframe.iloc[0: v_start_row - 1]
            df_right = movie_dataframe.iloc[v_start_row + v_size:]
            t_df = pd.concat([df_left, df_right], ignore_index=True)

        print("\n\n -- Fold " + str(i))
        print(" -- validation set index: " + str(v_df.index))
        print(" -- training set index: " + str(t_df.index))
        print(" -- training set index length: " + str(t_df.shape[0]))

        training_examples = preprocess_features(t_df)
        training_targets = preprocess_targets(t_df)

        validation_examples = preprocess_features(v_df)
        validation_targets = preprocess_targets(v_df)

        dnn_regressor, training_rmse, validation_rmse = train_predict(training_examples,
            training_targets, validation_examples, validation_targets, './model_' + str(i), None)
        validation_rmse_l.append(validation_rmse[-1])

    print(' ----- final k fold RMSLE are: ')
    print(validation_rmse_l)
    print(' -- final mean: ' + str(sum(validation_rmse_l) / k))
    print('\n\n\n')

def predict_test_set(k=10):
    test_movie_dataframe = pd.read_csv("test.csv")
    test_examples = preprocess_features(test_movie_dataframe)
    predict_test_input_fn = lambda: my_input_fn(test_examples, 
                                                  test_examples["budget"], 
                                                  num_epochs=1, 
                                                  shuffle=False)

    predict_results = [0] * test_movie_dataframe.shape[0]

    for i in range(k):
        dnn_regressor = tf.estimator.DNNRegressor(
          feature_columns=construct_feature_columns(test_examples),
          hidden_units=dnn_hidden_units,
          optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.001, l2_regularization_strength=0.005),
          model_dir=None,
          warm_start_from='./model_' + str(i))
        test_predictions = dnn_regressor.predict(input_fn=predict_test_input_fn)
        test_predictions = np.array([item['predictions'][0] for item in test_predictions])

        print(test_predictions)
        print("test_predictions len " + str(len(test_predictions)))
        for i in range(test_predictions.size):
            # average on k model predictions
            predict_results[i] += math.exp(test_predictions[i]) / k

    import csv
    csvfile = open('./submission.csv', 'wb')
    fieldnames = [
        'id',
        'revenue',
        ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(test_predictions.size):
        revenue = predict_results[i]
        writer.writerow({'id': test_movie_dataframe['id'][i], 'revenue': revenue})


from sklearn.metrics import mean_squared_error
def score(data, y):
    validation_res = pd.DataFrame(
    {"id": data["id"].values,
     "transactionrevenue": data["revenue"].values,
     "predictedrevenue": np.expm1(y)})

    validation_res = validation_res.groupby("id")["transactionrevenue", "predictedrevenue"].sum().reset_index()
    return np.sqrt(mean_squared_error(np.log1p(validation_res["transactionrevenue"].values), 
                                     np.log1p(validation_res["predictedrevenue"].values)))


from sklearn.model_selection import GroupKFold

class KFoldValidation():
    def __init__(self, data, n_splits=6):
        unique_vis = np.array(sorted(data['id'].astype(str).unique()))
        folds = GroupKFold(n_splits)
        ids = np.arange(data.shape[0])
        
        self.fold_ids = []
        for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
            self.fold_ids.append([
                    ids[data['id'].astype(str).isin(unique_vis[trn_vis])],
                    ids[data['id'].astype(str).isin(unique_vis[val_vis])]
                ])
            
    def validate(self, train, test, features, model, name="", prepare_stacking=False, 
                 fit_params={"early_stopping_rounds": 500, "verbose": 100, "eval_metric": "rmse"}):
        model.FI = pd.DataFrame(index=features)
        full_score = 0
        
        if prepare_stacking:
            test[name] = 0
            train[name] = np.NaN
        
        for fold_id, (trn, val) in enumerate(self.fold_ids):
            devel = train[features].iloc[trn]
            y_devel = np.log1p(train["revenue"].iloc[trn])
            valid = train[features].iloc[val]
            y_valid = np.log1p(train["revenue"].iloc[val])
                       
            print("Fold ", fold_id, ":")
            model.fit(devel, y_devel, eval_set=[(valid, y_valid)], **fit_params)
            
            if len(model.feature_importances_) == len(features):  
                model.FI['fold' + str(fold_id)] = model.feature_importances_ / model.feature_importances_.sum()

            predictions = model.predict(valid)
            predictions[predictions < 0] = 0
            print("Fold ", fold_id, " error: ", mean_squared_error(y_valid, predictions)**0.5)
            
            fold_score = score(train.iloc[val], predictions)
            full_score += fold_score / len(self.fold_ids)
            print("Fold ", fold_id, " score: ", fold_score)
            if prepare_stacking:
                train[name].iloc[val] = predictions
                
                test_predictions = model.predict(test[features])
                test_predictions[test_predictions < 0] = 0
                test[name] += test_predictions / len(self.fold_ids)
                
        print("Final score: ", full_score)
        return full_score


def convert_label_to_int(df, feature_name, vocabulary_list):
    def f(x):
        for i in range(len(vocabulary_list)):
            if vocabulary_list[i] == x:
                return i
            return -1
    df[feature_name] = df[feature_name].apply(lambda x : f(x))
    return df

def process_additiona_features(df):
    df.drop('imdb_id', axis=1, inplace=True)

    df['rating'] = df['rating'].fillna(df['rating'].mean())
    df['vote_count'] = df['totalVotes'].fillna(df['totalVotes'].mean())
    df.drop('totalVotes',  axis=1, inplace=True)

    #vote_count_na = df.groupby(["release_year","original_language"])['vote_count'].mean().reset_index()
    #df[df.vote_count.isna()]['vote_count'] = df.merge(vote_count_na, how = 'left' ,on = ["release_year","original_language"])

    df['meanruntimeByYear'] = df.groupby("release_year")["runtime"].aggregate('mean')
    df['meanPopularityByYear'] = df.groupby("release_year")["popularity"].aggregate('mean')
    df['meanBudgetByYear'] = df.groupby("release_year")["budget"].aggregate('mean')

    df['_popularity_totalVotes_ratio'] = df['vote_count']/df['popularity']
    df['_popularity2_totalVotes_ratio'] = df['vote_count']/df['popularity2']
    df['_totalVotes_releaseYear_ratio'] = df['vote_count']/df['release_year']
    df['_budget_totalVotes_ratio'] = df['budget']/df['vote_count']
    
    
    df['_rating_popularity_ratio'] = df['rating']/df['popularity']
    df['_rating_popularity2_ratio'] = df['rating']/df['popularity2']
    df['_rating_totalVotes_ratio'] = df['vote_count']/df['rating']
    df['_budget_rating_ratio'] = df['budget']/df['rating']
    df['_runtime_rating_ratio'] = df['runtime']/df['rating']

    df['popularity2'] = np.log1p(df['popularity2']).fillna(0)
    df['rating'] =  linear_process(df['rating']).fillna(0)
    df['vote_count'] = np.log1p(df['vote_count']).fillna(0)
    df.reset_index(inplace=True, drop=True)

    return df


def lgbm():
    import time
    random_seed = int(time.time()) 

    train = preprocess_features(movie_dataframe)
    train['id'] = movie_dataframe['id']
    train['revenue'] = movie_dataframe['revenue']
    convert_label_to_int(train, 'original_language', language_v_list)

    test_raw = pd.read_csv("test.csv")
    test = preprocess_features(test_raw)
    test['id'] = test_raw['id']
    convert_label_to_int(test, 'original_language', language_v_list)


    trainAdditionalFeatures = pd.read_csv('./TrainAdditionalFeatures.csv')[['imdb_id','popularity2','rating', 'totalVotes']]
    testAdditionalFeatures = pd.read_csv('./TestAdditionalFeatures.csv')[['imdb_id','popularity2','rating', 'totalVotes']]

    train = pd.merge(train, trainAdditionalFeatures, how='left', on=['imdb_id'])
    test = pd.merge(test, testAdditionalFeatures, how='left', on=['imdb_id'])

    train = process_additiona_features(train)
    test = process_additiona_features(test)

    features = list(train.columns)
    features =  [i for i in features if i != 'id' and i != 'revenue' and i != 'imdb_id']

    Kfolder = KFoldValidation(train)

    import lightgbm as lgb
    lgbmodel = lgb.LGBMRegressor(n_estimators=10000, 
                             objective='regression', 
                             metric='rmse',
                             max_depth = 5,
                             num_leaves=30, 
                             min_child_samples=100,
                             learning_rate=0.01,
                             boosting = 'gbdt',
                             min_data_in_leaf= 10,
                             feature_fraction = 0.9,
                             bagging_freq = 1,
                             bagging_fraction = 0.9,
                             importance_type='gain',
                             lambda_l1 = 0.2,
                             bagging_seed=random_seed, 
                             subsample=.8, 
                             colsample_bytree=.9,
                             use_best_model=True)
    Kfolder.validate(train, test, features , lgbmodel, name="lgbfinal", prepare_stacking=True)

    test['revenue'] =  np.expm1(test["lgbfinal"])
    test[['id','revenue']].to_csv('submission_lgb.csv', index=False)
    test[['id','revenue']].head()

    import xgboost as xgb
    xgbmodel = xgb.XGBRegressor(max_depth=5, 
                            learning_rate=0.01, 
                            n_estimators=10000, 
                            objective='reg:linear', 
                            gamma=1.45, 
                            seed=random_seed, 
                            silent=True,
                            subsample=0.8, 
                            colsample_bytree=0.7, 
                            colsample_bylevel=0.5)
    Kfolder.validate(train, test, features, xgbmodel, name="xgbfinal", prepare_stacking=True)

    test['revenue'] =  np.expm1(test["xgbfinal"])
    test[['id','revenue']].to_csv('submission_xgb.csv', index=False)
    test[['id','revenue']].head()

    import catboost as cat
    catmodel = cat.CatBoostRegressor(iterations=10000, 
                                 learning_rate=0.01, 
                                 depth=5, 
                                 eval_metric='RMSE',
                                 colsample_bylevel=0.8,
                                 bagging_temperature = 0.2,
                                 metric_period = None,
                                 early_stopping_rounds=200,
                                 random_seed=random_seed)
    Kfolder.validate(train, test, features , catmodel, name="catfinal", prepare_stacking=True,
               fit_params={"use_best_model": True, "verbose": 100})

    test['revenue'] =  np.expm1(test["catfinal"])
    test[['id','revenue']].to_csv('submission_cat.csv', index=False)
    test[['id','revenue']].head()

    test['revenue'] =  np.expm1(0.4 * test["lgbfinal"]+ 0.4 * test["catfinal"] + 0.2 * test["xgbfinal"])
    test[['id','revenue']].to_csv('submission_Dragon1.csv', index=False)
    test[['id','revenue']].head()

    test['revenue'] =  np.expm1((test["lgbfinal"] + test["catfinal"] + test["xgbfinal"])/3)
    test[['id','revenue']].to_csv('submission_Dragon2.csv', index=False)
    test[['id','revenue']].head()


    train['Revenue_lgb'] = train["lgbfinal"]

    print("RMSE model lgb :" ,score(train, train.Revenue_lgb),)

    train['Revenue_xgb'] = train["xgbfinal"]

    print("RMSE model xgb :" ,score(train, train.Revenue_xgb))

    train['Revenue_cat'] = train["catfinal"]

    print("RMSE model cat :" ,score(train, train.Revenue_cat))

    train['Revenue_Dragon1'] = 0.4 * train["lgbfinal"] + \
                                   0.2 * train["xgbfinal"] + \
                                   0.4 * train["catfinal"]

    print("RMSE model Dragon1 :" ,score(train, train.Revenue_Dragon1))

    train['Revenue_Dragon2'] = 0.35 * train["lgbfinal"] + \
                                   0.3 * train["xgbfinal"] + \
                                   0.35 * train["catfinal"]

    print("RMSE model Dragon2 :" ,score(train, train.Revenue_Dragon2))

if __name__ == "__main__":
    # k = 20
    # k_fold(k)
    # predict_test_set(k)
    lgbm()

'''
('RMSE model lgb :', 1.8380955518206552)
('RMSE model xgb :', 1.8472091729832854)
('RMSE model cat :', 1.8075541092962701)
('RMSE model Dragon1 :', 1.8065819168154829)
('RMSE model Dragon2 :', 1.809241445818128)

test RMSE: 1.88
'''

