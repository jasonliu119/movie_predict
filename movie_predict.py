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

movie_dataframe = movie_dataframe.reindex(np.random.permutation(movie_dataframe.index))

# budget: + 1 and then log norm
# genres: one-hot encoding, categorical_column_with_vocabulary_list
# homepage: has or not
# 

def describe_lang():
    value_dict = {}
    for l in movie_dataframe['original_language']:
        try:
            if l in value_dict:
                value_dict[l] += 1
            else:
                value_dict[l] = 1
        except:
            pass
    print(value_dict.keys())
    print(len(value_dict))
    print(len(np.array(movie_dataframe['original_language'])))

def get_top_elements(feature_name, length = 30, field = 'name'):
    gen_dict = {}
    null_count = 0
    for s in movie_dataframe[feature_name]:
        # print(s)
        try:
            j = json.loads(str(s).replace('\'', '\"'))

            for gen in j:
                type_name = gen[field]
                if type_name in gen_dict:
                    gen_dict[type_name] += 1
                else:
                    gen_dict[type_name] = 1
        except:
            null_count += 1
    top_l = sorted(gen_dict.items(), key=lambda x : x[1], reverse=True)[:length]
    sorted_dict = dict(top_l)
    print(" -- Top " + feature_name)
    print(top_l)
    print("vocabulary_list: " + str(sorted_dict.keys()))
    return sorted_dict.keys()

def f(x):
    ret = []
    if pd.isna(x):
        return ret
    print(x)
    for i in range(len(x)):
        print " -- element: " + str(x[i])
        ret.append(x[i]['name'])
    return ret

import ast
dict_columns = ['belongs_to_collection', 'genres', 'production_companies',
                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']

def text_to_dict(df):
    for column in dict_columns:
        df[column] = df[column].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x) )
    return df

print "Converting text in original dataframe into dict with ast.literal_eval...\n\n"
to_dict_movie_dataframe = movie_dataframe.copy()
to_dict_movie_dataframe = text_to_dict(to_dict_movie_dataframe)

# use lambda to get the top list of a certain feature after converting the text to dict
def get_top(feature_name, length = 30, field = 'name'):
    x = to_dict_movie_dataframe[feature_name].apply(lambda x : [x[i][field] for i in range(len(x))] if x != {} else []).values
    top = [x[0] for x in Counter([i for j in x for i in j]).most_common(length)]
    print " -- Top " + feature_name + "\n" + str(top)
    print '\n\n'
    return top

def describe_json_feature(feature_name, field = 'name'):
    gen_dict = {}
    null_count = 0
    for s in movie_dataframe[feature_name]:
        print(s)
        try:
            j = json.loads(str(s).replace('\'', '\"'))

            for gen in j:
                type_name = gen[field]
                if type_name in gen_dict:
                    gen_dict[type_name] += 1
                else:
                    gen_dict[type_name] = 1
        except:
            null_count += 1
    sorted_dict = dict(sorted(gen_dict.items(), key=lambda x : x[1], reverse=True)[:20])
    print(sorted_dict)
    print(sorted_dict.keys())
    print("nan count " + str(null_count))

def preprocess_targets(movie_dataframe):
  output_targets = pd.DataFrame()
  output_targets["revenue"] = np.log1p(movie_dataframe["revenue"])
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

company_v_list = get_top('production_companies')

language_v_list = get_top('spoken_languages', 15, 'iso_639_1')

keyword_v_list = get_top('Keywords', 30)

cast_v_list = get_top('cast', 30)

crew_v_list = get_top('crew', 30)

collection_v_list = get_top('belongs_to_collection', 30)

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
     ]]

    # pf means processed features
    pf = selected_features.copy()

    pf["budget"] = np.log1p(pf["budget"])

    pf = convert_to_multi_hot_column(pf, "genres", genres_v_list)

    pf["homepage"] = (pf["homepage"].isnull() == True).astype('int')

    pf["popularity"] = np.log1p(pf["popularity"])

    # pf["release_date"] = pf["release_date"].apply(lambda x : date(x))
    pf['release_day'] = pf['release_date'].apply(lambda x: date(x, 'weekday')).astype(int)
    pf['release_month'] = pf['release_date'].apply(lambda x: date(x, 'month')).astype(int)
    pf['release_year'] = pf['release_date'].apply(lambda x: date(x, 'year'))
    pf['_budget_year_ratio'] = dataframe['budget'].fillna(0) / (pf['release_year'] * pf['release_year'])

    # the count of the movies that was released in the same month
    # pf['release_month_year_count'] = get_release_month_year_count(pf)

    pf['release_year'] = linear_process(pf['release_year'])

    pf = convert_to_multi_hot_column(pf, "production_companies", company_v_list)

    pf['runtime'] = linear_process(pf['runtime']).fillna(0)

    pf = convert_to_multi_hot_column(pf, "spoken_languages", language_v_list)

    pf = convert_to_multi_hot_column(pf, "Keywords", keyword_v_list)

    pf = convert_to_multi_hot_column(pf, "cast", cast_v_list)

    pf = convert_to_multi_hot_column(pf, "crew", crew_v_list)

    pf = convert_to_multi_hot_column(pf, "belongs_to_collection", collection_v_list)

    # for k in pf:
    #     print(k + " --> " + str(pf[k].size))

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

training_rate = 0.8
validation_rate = 1 - training_rate
training_number = int(movie_dataframe.shape[0] * training_rate)
validation_number = movie_dataframe.shape[0] - training_number

training_examples = preprocess_features(movie_dataframe.head(training_number))
training_targets = preprocess_targets(movie_dataframe.head(training_number))

validation_examples = preprocess_features(movie_dataframe.tail(validation_number))
validation_targets = preprocess_targets(movie_dataframe.tail(validation_number))

def train_nn_regression_model(
    my_optimizer,
    steps,
    batch_size,
    hidden_units,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
  periods = 10
  steps_per_period = steps / periods
  
  # Create a DNNRegressor object.
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  dnn_regressor = tf.estimator.DNNRegressor(
      feature_columns=construct_feature_columns(training_examples),
      hidden_units=hidden_units,
      optimizer=my_optimizer
  )
  
  # Create input functions.
  training_input_fn = lambda: my_input_fn(training_examples, 
                                          training_targets["revenue"], 
                                          batch_size=batch_size)
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
    # Occasionally print the current loss.
    print("  period %02d : %0.2f" % (period, training_root_mean_squared_log_error))
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

def train_predict():
    _ = train_nn_regression_model(
        my_optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.001, l2_regularization_strength=0.001),
        steps=20000,
        batch_size=100,
        hidden_units=[512, 256],
        training_examples=training_examples,
        training_targets=training_targets,
        validation_examples=validation_examples,
        validation_targets=validation_targets)

if __name__ == "__main__":
    train_predict()
    

