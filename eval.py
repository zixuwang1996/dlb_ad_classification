#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
from sklearn.metrics import precision_recall_fscore_support
from data_helpers import CRISDataset

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("data_path", "./data", "Data source for the negative data.")
tf.flags.DEFINE_string("positive_data_file", "./data/cris-new/test.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/cris-new/test.neg", "Data source for the negative data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")
tf.flags.DEFINE_float("temperature", 1.0, "calibration scalar")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

def softmax(x):
  """Compute softmax values for each sets of scores in x."""
  return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


FLAGS = tf.flags.FLAGS
#FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparation
print ("Loading Dataset ...")
vocab_path = os.path.join(FLAGS.data_path, "vocab.pkl")
dataset = CRISDataset(FLAGS.data_path, vocab_path, 5000, test=True)
X = dataset.load()
print ("Dataset loaded. Preparing data and loading embeddings ...")

x_test, y_test = X, None

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        scores = graph.get_operation_by_name("output/scores").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []
        all_scores = []

        for x_test_batch in batches:
            batch_predictions, batch_scores = sess.run([predictions, scores], {input_x: x_test_batch, dropout_keep_prob: 1.0})

            if len(all_predictions) == 0:
              all_predictions = batch_predictions
            else:
              all_predictions = np.concatenate([all_predictions, batch_predictions])
            #batch_scores = np.amax(all_scores, axis=1) / FLAGS.temperature
            if len(all_scores) == 0:
              all_scores = batch_scores
            else:
              all_scores = np.concatenate([all_scores, batch_scores], axis=-1)


if True:
    #print("Confidence: {}".format(np.mean(all_scores)))
    #y_test = np.argmax(y_test, axis=1)
    #correct_predictions = float(sum(all_predictions == y_test))
    #score = precision_recall_fscore_support(y_test, all_predictions, average='binary')

    print("Predictions: {}".format(softmax(all_scores)[:,1]))
    #print("True labels: {}".format(y_test))
    # print("Precision: {}, Recall: {}, F-score: {}".format(score[0], score[1], score[2]))
    # print("Total number of test examples: {}".format(len(y_test)))
#    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

# Save the evaluation to a csv
predictions_human_readable = softmax(all_scores)[:,1]
out_path = os.path.join(FLAGS.checkpoint_dir, "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
np.savetxt(out_path, predictions_human_readable, delimiter=",", fmt='%10.5f')
