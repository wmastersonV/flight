#!/usr/bin/env python

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

airlines = ["AS", "DL", "B6", "KH", "US", "MQ", "CO", "VX", "OO", "HA", "NW", "WN", "XE", "AA", "FL", "UA", "YV", "EV", "OH", "F9", "TZ", "9E", "DH", "HP"]
depart_air = ["ONT", "RNO", "GST", "PSG", "KTN", "HPN", "ISP", "GCC", "AEX", "TUS", "BOI", "BIL", "LIH", "MRY", "SDF", "SBA", "OKC", "TLH", "WRG", "VIS", "TUL", "STT", "EAU", "RDM", "GRR", "TYS", "CAK", "PSP", "ITO", "HLN", "SCC", "CDV", "MLI", "JAC", "STX", "BLI", "BRW", "MFR", "MAZ", "OME", "MAF", "MKG", "PSC", "JAN", "CRP", "MEI", "CDC", "INL", "PNS", "ESC", "LWB", "GSP", "PFN", "GPT", "PIR", "AUS", "BNA", "BWI", "CLE", "CLT", "DFW", "DTW", "EWR", "FLL", "HNL", "IAD", "IAH", "IND", "LAS", "LAX", "LGA", "MCI", "MIA", "OAK", "ORD", "PDX", "RDU", "TPA", "ALB", "MHT", "DAY", "AMA", "BFL", "GNV", "CHS", "SAV", "EYW", "GSO", "EGE", "OTH", "PWM", "MYR", "AVP", "BTV", "SGF", "CID", "ICT", "DAB", "CRW", "MMH", "CHA", "CAE", "VPS", "ART", "LNK", "MBS", "LSE", "ASE", "MTJ", "ORH", "SPI", "AZO", "DLH", "SUX", "RST", "CHO", "GJT", "FWA", "CMX", "CMI", "AVL", "CPR", "PSE", "GUC", "CLL", "MLB", "FSM", "DRO", "SJT", "TXK", "LAW", "BRO", "ILE", "MVY", "BQN", "ABR", "ISN", "MOT", "GFK", "ACY", "ILG", "DHN", "VLD", "ISO", "TTN", "TUP", "APF", "HTS", "SOP", "OAJ", "FLO", "MCN", "MTH", "EWN", "HKY", "PLN", "EFD", "VCT", "HVN", "GUM", "MWH", "SLE", "SUN", "WYS", "MOD", "RDD", "PMD", "LAR", "OXR", "SMX", "PUB", "CKB", "FMN", "PVU", "BFF", "RKS", "OGG", "RSW", "PBI", "JAX", "CEC", "ANC", "BDL", "SJU", "PVD", "SYR", "MLU", "FAT", "MKE", "BET", "LNY", "ADK", "MSO", "BHM", "LFT", "HIB", "GTF", "GEG", "KOA", "FNT", "BUF", "ELP", "SWF", "OMA", "RIC", "ORF", "SIT", "YAK", "JNU", "LGB", "EKO", "COS", "ROC", "CWA", "LBB", "AKN", "IMT", "SGU", "BMI", "OTZ", "MKK", "ACV", "FAI", "ROA", "EVV", "ADQ", "DLG", "GTR", "PIE", "BGR", "ILM", "BTR", "PIA", "DUT", "SHV", "HSV", "ABQ", "ATL", "BOS", "CMH", "CVG", "DAL", "DCA", "DEN", "HOU", "JFK", "MCO", "MDW", "MEM", "MSP", "MSY", "PHL", "PHX", "PIT", "SAN", "SAT", "SEA", "SFO", "SJC", "SLC", "SMF", "SNA", "STL", "BUR", "LIT", "FSD", "SBP", "MSN", "DSM", "FAR", "EUG", "XNA", "LMT", "MDT", "ABE", "ERI", "SRQ", "SBN", "LEX", "GRB", "IPL", "TVC", "PAH", "MOB", "BIS", "LAN", "ATW", "ALO", "TRI", "TOL", "RAP", "BZN", "MHK", "DBQ", "HDN", "HRL", "PHF", "ACT", "ABI", "SAF", "FAY", "AGS", "BPT", "GGG", "SPS", "ROW", "MGM", "CSG", "CYS", "TYR", "LRD", "GRK", "GRI", "MFE", "JLN", "GCK", "LCH", "BGM", "ACK", "IDA", "RHI", "APN", "BRD", "BJI", "LYH", "BQK", "COU", "ABY", "ITH", "ELM", "HOB", "DRT", "BTM", "TWF", "LWS", "COD", "PIH", "YKM", "OGD", "CIC", "RFD", "IYK", "FLG", "TEX", "SHD", "MKC", "GLH"]

arrival_air = ["GCC", "ISP", "BOI", "JAN", "VIS", "PNS", "SBA", "MEI", "INL", "VLD", "ITO", "PSP", "ONT", "TUL", "KTN", "RNO", "MFR", "EAU", "LIH", "MRY", "MAF", "ESC", "CDC", "OME", "TLH", "STX", "HPN", "BIL", "BRW", "GRR", "LWB", "OKC", "DAY", "CAK", "GPT", "TUS", "WRG", "SGF", "STT", "SCC", "SDF", "GST", "AEX", "HLN", "JAC", "BTV", "PSG", "CDV", "MKG", "MCN", "BRO", "MLI", "AVL", "MAZ", "SAV", "GSO", "PFN", "ALB", "PWM", "TUP", "TYS", "ICT", "ILG", "CRP", "LAW", "SOP", "TTN", "ASE", "MTJ", "PSE", "AVP", "DAB", "APF", "CHO", "GNV", "VPS", "EWN", "GSP", "DHN", "EGE", "MHT", "MTH", "ACY", "FWA", "CHA", "CID", "ISO", "EYW", "FLO", "LNK", "HKY", "FSM", "CRW", "OAJ", "CAE", "MLB", "HTS", "AZO", "BQN", "CMI", "HVN", "RDM", "LAR", "BLI", "BFL", "DRO", "GUC", "PSC", "GJT", "GFK", "ISN", "MOT", "CPR", "SJT", "TXK", "ILE", "CLL", "SPI", "RST", "DLH", "LSE", "MBS", "PLN", "GUM", "EFD", "VCT", "MVY", "OXR", "SMX", "MOD", "ABR", "PIR", "SUX", "CMX", "ART", "ORH", "OTH", "MMH", "MWH", "RDD", "PMD", "SUN", "WYS", "SLE", "LAS", "MIA", "PDX", "CLE", "ORD", "FLL", "DFW", "CHS", "OAK", "AUS", "DTW", "CLT", "AMA", "IND", "IAH", "MCI", "BNA", "HNL", "RDU", "EWR", "IAD", "TPA", "LAX", "LGA", "BWI", "MYR", "CKB", "CBM", "PVU", "KOA", "ACV", "ADQ", "AKN", "ORF", "PBI", "JNU", "BDL", "PVD", "RKS", "OMA", "ROC", "GTR", "SRQ", "MKK", "GTF", "HSV", "MSO", "BET", "COS", "RSW", "LGB", "BHM", "LBB", "RIC", "SHV", "FAT", "OGG", "SJU", "CWA", "SYR", "DLG", "EKO", "BTR", "DSM", "GRB", "LEX", "GEG", "ATW", "FNT", "YAK", "ANC", "MGM", "RAP", "DUT", "BMI", "COD", "ADK", "FAI", "SIT", "OTZ", "MSN", "EUG", "ELP", "JAX", "MLU", "SWF", "PIE", "LFT", "SGU", "HIB", "PHF", "IMT", "ROA", "BUR", "XNA", "PIA", "AGS", "LAN", "MDT", "BGM", "TVC", "FSD", "CSG", "BPT", "BGR", "COU", "LIT", "BQK", "ABY", "LYH", "MFE", "BZN", "TRI", "ILM", "ERI", "ABE", "HDN", "FAY", "MOB", "SBN", "EVV", "TOL", "GRK", "HRL", "ACK", "RFD", "BIS", "FAR", "IDA", "GGG", "ABI", "SPS", "ROW", "GCK", "CYS", "LCH", "JLN", "ACT", "MHK", "LRD", "TYR", "GRI", "SAF", "ITH", "APN", "ELM", "LNY", "HOB", "DRT", "SBP", "IYK", "IPL", "SHD", "BRD", "ALO", "BJI", "RHI", "DBQ", "PAH", "LMT", "TEX", "FLG", "CIC", "TWF", "YKM", "LWS", "BTM", "PIH", "PIT", "STL", "CMH", "SAN", "MSY", "SAT", "JFK", "MSP", "ATL", "MEM", "ABQ", "CVG", "PHL", "SEA", "SLC", "HOU", "MDW", "CEC", "BOS", "SJC", "SFO", "DEN", "SMF", "MKE", "BUF", "DCA", "PHX", "MCO", "DAL", "SNA", "GLH", "MKC"]
all_air = list(set(depart_air + arrival_air))

CSV_COLUMNS = ['departure_lat', 'departure_lon', 'arrival_lat', 'arrival_lon', 'airline', 'departure_airport','arrival_airport','dow','week','month', \
               'arrival_delay', 'delay_0', 'delay_15', 'delay_30', 'delay_45', 'delay_60', 'depart_minutes', 'scheduled_flight_time']

LABEL_COLUMN = 'arrival_delay'

DEFAULTS = [[999.0], [999.0], [999.0], [999.0], ['NA'], ['NA'],['NA'], [999], [999], [999], [6.0], [2], [2], [2], [2], [2], [807.0], [111.0]]

# These are the raw input columns, and will be provided for prediction also
INPUT_COLUMNS = [
    tf.feature_column.categorical_column_with_identity('week', num_buckets=54),
    tf.feature_column.categorical_column_with_identity('dow', num_buckets = 8),
    tf.feature_column.categorical_column_with_identity('month', num_buckets = 13),
    tf.feature_column.categorical_column_with_vocabulary_list('airline', vocabulary_list=airlines),
    tf.feature_column.categorical_column_with_vocabulary_list('arrival_airport',vocabulary_list=all_air),
    tf.feature_column.categorical_column_with_vocabulary_list('departure_airport',vocabulary_list=all_air),
    tf.feature_column.numeric_column('depart_minutes'),
    tf.feature_column.numeric_column('scheduled_flight_time'),
    tf.feature_column.numeric_column('departure_lat'),
    tf.feature_column.numeric_column('departure_lon'),
    tf.feature_column.numeric_column('arrival_lat'),
    tf.feature_column.numeric_column('arrival_lon'),
    
#    engineered features
    tf.feature_column.numeric_column('latdiff'),
    tf.feature_column.numeric_column('londiff'),
    tf.feature_column.numeric_column('euclidean')
]


# Build the estimator
def build_estimator(model_dir, nbuckets, hidden_units):
    """
     Build an estimator starting from INPUT COLUMNS.
     These include feature transformations and synthetic features.
     The model is a wide-and-deep model.
  """

    (week, dow, month, airline, arrival_airport, departure_airport, depart_minutes, scheduled_flight_time, departure_lat, departure_lon, \
    arrival_lat, arrival_lon, latdiff, londiff, euclidean) = INPUT_COLUMNS

    # Bucketize the lats & lons
    latbuckets = np.linspace(-180, 180, nbuckets).tolist()
    lonbuckets = np.linspace(-180, 180, nbuckets).tolist()
    b_plat = tf.feature_column.bucketized_column(departure_lat, latbuckets)
#    print(b_plat)
    b_dlat = tf.feature_column.bucketized_column(arrival_lat, latbuckets)
    b_plon = tf.feature_column.bucketized_column(departure_lon, lonbuckets)
    b_dlon = tf.feature_column.bucketized_column(arrival_lon, lonbuckets)

    # Feature cross

    ploc = tf.feature_column.crossed_column([b_plat, b_plon], 100 * 100)
    dloc = tf.feature_column.crossed_column([b_dlat, b_dlon], 100 * 100)
    pd_pair = tf.feature_column.crossed_column([ploc, dloc], 100 ** 4 )
#    day_hr =  tf.feature_column.crossed_column([dow, tf.floor(depart_minutes/60)], 24 * 7)

    # Wide columns and deep columns.
    wide_columns = [
        # Feature crosses
        dloc, ploc, pd_pair,
        #day_hr,

        # Sparse columns
        week, dow, month, airline 
        #arrival_airport, departure_airport

        # Anything with a linear relationship
#        pcount 
    ]

    deep_columns = [
        # Embedding_column to "group" together ...
        tf.feature_column.embedding_column(pd_pair, 10),
#        tf.feature_column.embedding_column(day_hr, 10),

        # Numeric columns
        depart_minutes, scheduled_flight_time, departure_lat, departure_lon,
        arrival_lat, arrival_lon,
        latdiff, londiff, euclidean
    ]
    
    estimator = tf.estimator.DNNLinearCombinedRegressor(
        model_dir = model_dir,
        linear_feature_columns = wide_columns,
        dnn_feature_columns = deep_columns,
        dnn_hidden_units = hidden_units or [128, 32, 4])

    # add extra evaluation metric for hyperparameter tuning
    estimator = tf.contrib.estimator.add_metrics(estimator, add_eval_metrics)
    return estimator

# Create feature engineering function that will be used in the input and serving input functions
def add_engineered(features):
    # this is how you can do feature engineering in TensorFlow
    lat1 = features['departure_lat']
    lat2 = features['arrival_lat']
    lon1 = features['departure_lon']
    lon2 = features['arrival_lon']
    latdiff = (lat1 - lat2)
    londiff = (lon1 - lon2)
    
    # set features for distance with sign that indicates direction
    features['latdiff'] = latdiff
    features['londiff'] = londiff
    dist = tf.sqrt(latdiff * latdiff + londiff * londiff)
    features['euclidean'] = dist
    return features

# Create serving input function to be able to serve predictions
def serving_input_fn():
#     pb.set_trace()
#         (week, dow, month, airline, arrival_airport, departure_airport, depart_minutes, scheduled_flight_time, departure_lat, departure_lon, \
#     arrival_lat, arrival_lon, latdiff, londiff, euclidean)

#     tf.feature_column.categorical_column_with_identity('week', num_buckets=54),
#     tf.feature_column.categorical_column_with_identity('dow', num_buckets = 8),
#     tf.feature_column.categorical_column_with_identity('month', num_buckets = 13),
#     tf.feature_column.categorical_column_with_vocabulary_list('airline', vocabulary_list=airlines),
#     tf.feature_column.categorical_column_with_vocabulary_list('arrival_airport',vocabulary_list=all_air),
#     tf.feature_column.categorical_column_with_vocabulary_list('departure_airport',vocabulary_list=all_air),
#     tf.feature_column.numeric_column('depart_minutes'),
#     tf.feature_column.numeric_column('scheduled_flight_time'),
#     tf.feature_column.numeric_column('departure_lat'),
#     tf.feature_column.numeric_column('departure_lon'),
#     tf.feature_column.numeric_column('arrival_lat'),
#     tf.feature_column.numeric_column('arrival_lon'),
    
# #    engineered features
#     tf.feature_column.numeric_column('latdiff'),
#     tf.feature_column.numeric_column('londiff'),
#     tf.feature_column.numeric_column('euclidean')

    feature_placeholders = {
        # All the real-valued columns
        column.name: tf.placeholder(tf.float32, [None]) for column in INPUT_COLUMNS
    }
    feature_placeholders['airline'] = tf.placeholder(tf.string, [None])
    feature_placeholders['arrival_airport'] = tf.placeholder(tf.string, [None])
    feature_placeholders['departure_airport'] = tf.placeholder(tf.string, [None])
    feature_placeholders['dow'] = tf.placeholder(tf.int64, [None])
    feature_placeholders['week'] = tf.placeholder(tf.int64, [None])
    feature_placeholders['month'] = tf.placeholder(tf.int64, [None])

    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_placeholders.items()
    }
    return tf.estimator.export.ServingInputReceiver(add_engineered(features), feature_placeholders)

# Create input function to load data into datasets
def read_dataset(filename, mode, batch_size = 512):
    def _input_fn():
        def decode_csv(value_column):
            columns = tf.decode_csv(value_column, record_defaults = DEFAULTS)
            features = dict(zip(CSV_COLUMNS, columns))
            label = features.pop(LABEL_COLUMN)
            return add_engineered(features), label
        
        # Create list of files that match pattern
        file_list = tf.gfile.Glob(filename)

        # Create dataset from file list
        dataset = tf.data.TextLineDataset(file_list).map(decode_csv)

        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None # indefinitely
            dataset = dataset.shuffle(buffer_size = 10 * batch_size)
        else:
            num_epochs = 1 # end-of-input after this

        dataset = dataset.repeat(num_epochs).batch(batch_size)
        
        batch_features, batch_labels = dataset.make_one_shot_iterator().get_next()
        return batch_features, batch_labels
    return _input_fn

# Create estimator train and evaluate function
def train_and_evaluate(args):
    estimator = build_estimator(args['output_dir'], args['nbuckets'], args['hidden_units'].split(' '))
    train_spec = tf.estimator.TrainSpec(
        input_fn = read_dataset(
            filename = args['train_data_paths'],
            mode = tf.estimator.ModeKeys.TRAIN,
            batch_size = args['train_batch_size']),
        max_steps = args['train_steps'])
    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(
        input_fn = read_dataset(
            filename = args['eval_data_paths'],
            mode = tf.estimator.ModeKeys.EVAL,
            batch_size = args['eval_batch_size']),
        steps = 100,
        exporters = exporter)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

# If we want to use TFRecords instead of CSV
def gzip_reader_fn():
    return tf.TFRecordReader(options=tf.python_io.TFRecordOptions(
            compression_type = tf.python_io.TFRecordCompressionType.GZIP))

def generate_tfrecord_input_fn(data_paths, num_epochs = None, batch_size = 512, mode = tf.estimator.ModeKeys.TRAIN):
    def get_input_features():
        # Read the tfrecords. Same input schema as in preprocess
        input_schema = {}
        if mode != tf.estimator.ModeKeys.INFER:
            input_schema[LABEL_COLUMN] = tf.FixedLenFeature(shape = [1], dtype = tf.float32, default_value = 0.0)
        for name in ['dayofweek', 'key']:
            input_schema[name] = tf.FixedLenFeature(shape = [1], dtype = tf.string, default_value = 'null')
        for name in ['hourofday']:
            input_schema[name] = tf.FixedLenFeature(shape = [1], dtype = tf.int64, default_value = 0)
        for name in SCALE_COLUMNS:
            input_schema[name] = tf.FixedLenFeature(shape = [1], dtype = tf.float32, default_value = 0.0)

        # How? 
        keys, features = tf.contrib.learn.io.read_keyed_batch_features(
            data_paths[0] if len(data_paths) == 1 else data_paths,
            batch_size,
            input_schema,
            reader = gzip_reader_fn,
            reader_num_threads = 4,
            queue_capacity = batch_size * 2,
            randomize_input = (mode != tf.estimator.ModeKeys.EVAL),
            num_epochs = (1 if mode == tf.estimator.ModeKeys.EVAL else num_epochs))
        target = features.pop(LABEL_COLUMN)
        features[KEY_FEATURE_COLUMN] = keys
        return add_engineered(features), target

    # Return a function to input the features into the model from a data path.
    return get_input_features

def add_eval_metrics(labels, predictions):
    pred_values = predictions['predictions']
    return {
        'rmse': tf.metrics.root_mean_squared_error(labels, pred_values)
    }

