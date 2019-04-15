from __future__ import print_function

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

tf.logging.set_verbosity(tf.logging.ERROR)

movie_dataframe = pd.read_csv("train.csv")

movie_dataframe = movie_dataframe.reindex(np.random.permutation(movie_dataframe.index))

print(movie_dataframe['original_language'].describe())

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
    print(value_dict)
    print(len(value_dict))
    print(len(np.array(movie_dataframe['original_language'])))

def describe_genres():
    gen_dict = {}
    null_count = 0
    for s in movie_dataframe['genres']:
        print(s)
        try:
            j = json.loads(str(s).replace('\'', '\"'))

            for gen in j:
                type_name = gen['name']
                if type_name in gen_dict:
                    gen_dict[type_name] += 1
                else:
                    gen_dict[type_name] = 1
        except:
            null_count += 1
    print(gen_dict.keys())
    print("nan count " + str(null_count))

def preprocess_targets(california_housing_dataframe):
  """Prepares target features (i.e., labels) from California housing data set.

  Args:
    california_housing_dataframe: A Pandas DataFrame expected to contain data
      from the California housing data set.
  Returns:
    A DataFrame that contains the target feature.
  """
  output_targets = pd.DataFrame()
  # Scale the target to be in units of thousands of dollars.
  output_targets["median_house_value"] = (
    california_housing_dataframe["median_house_value"] / 1000.0)
  return output_targets

def process_budget(raw_series):
    # log and normalize
    series = raw_series.apply(lambda x:math.log(x + 1))
    min_val = series.min()
    max_val = series.max()
    scale = (max_val - min_val) / 2.0
    return series.apply(lambda x:((x - min_val) / scale) - 1.0)

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

def preprocess_features(movie_dataframe):
    selected_features = movie_dataframe[
    [
    "budget",
    "genres",
    "homepage",
    "original_language",
     ]]

    processed_features = selected_features.copy()
    processed_features["budget"] = process_budget(processed_features["budget"])
    processed_features["genres"] = process_genres(processed_features["genres"])
    processed_features["homepage"] = (processed_features["homepage"].isnull() == True).astype('int')
    return processed_features

def construct_feature_columns(input_features):
    ret = set()

    ret.add(tf.feature_column.numeric_column('budget'))

    genres = tf.feature_column.categorical_column_with_vocabulary_list(
    'genres', [u'Mystery', u'Romance', u'Family', u'Science Fiction', u'Horror',
        u'Crime', u'Drama', u'Fantasy', u'Western', u'Animation', u'Music',
        u'Adventure', u'Foreign', u'Action', u'TV Movie', u'Comedy', u'Documentary',
        u'War', u'Thriller', u'History'])
    ret.add(tf.feature_column.indicator_column(genres))

    ret.add(tf.feature_column.numeric_column('homepage'))

    original_language = tf.feature_column.categorical_column_with_identity(
        key='original_language',
        num_buckets=36)
    ret.add(tf.feature_column.indicator_column(original_language))

    return ret

def my_input_fn(df_features, targets, batch_size=1, shuffle=True, num_epochs=None):    
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}  
    
    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(10000)
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels
