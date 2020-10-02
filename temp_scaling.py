import os
import pdb
import argparse
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow_probability as tfp
import data_helpers
from tensorflow.contrib import learn
from text_cnn import TextCNN
from calibration_metrics import expected_calibration_error
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from data_helpers import CRISDataset
from sklearn.model_selection import train_test_split


slim = tf.contrib.slim

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size',
    type=int,
    default=16,
    help="batch size.")
parser.add_argument('--fold',
    type=int,
    default=0,
    help="fold.")

parser.add_argument('--save-dir',
    type=str,
    default='./log',
    help="Where to save the models.")
parser.add_argument('--data_path',
    type=str,
    default='./data/cris-new',
    help="Where the data are saved")
parser.add_argument('--checkpoint_dir',
    type=str,
    default='',
#        default='./runs/1580480137/checkpoints',
    help="Where the checkpoint are saved")

args, unparsed = parser.parse_known_args()


# Data Preparation
print ("Loading Dataset ...")
vocab_path = args.data_path + "/vocab.pkl"
dataset = CRISDataset(args.data_path, vocab_path, 5000)
X, Y = dataset.load()

np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(Y)))
x_shuffled = X[shuffle_indices]
y_shuffled = Y[shuffle_indices]
y_shuffled = np.argmax(y_shuffled, axis=1)

def temp_scaling(logits_nps, labels_nps, sess, maxiter=50):

    temp_var = tf.get_variable("temp", shape=[1], initializer=tf.initializers.constant(1.5))

    logits_tensor = tf.constant(logits_nps, name='logits_valid')
    labels_tensor = tf.constant(labels_nps, name='labels_valid')

    org_nll_loss_op = tf.losses.sparse_softmax_cross_entropy(
        labels=labels_tensor, logits=logits_tensor)

    ece = tfp.stats.expected_calibration_error(10, logits_tensor, labels_tensor, name='ece')

    acc_op = tf.metrics.accuracy(labels_tensor, tf.argmax(logits_tensor, axis=1))

    #########################################################################################
    logits_w_temp = tf.divide(logits_tensor, temp_var)

    # loss
    nll_loss_op = tf.losses.sparse_softmax_cross_entropy(
        labels=labels_tensor, logits=logits_w_temp)

    # optimizer
    optim = tf.contrib.opt.ScipyOptimizerInterface(nll_loss_op, options={'maxiter': maxiter})

    sess.run(temp_var.initializer)
    sess.run(tf.local_variables_initializer())
    org_nll_loss = sess.run(org_nll_loss_op)

    optim.minimize(sess)

    nll_loss = sess.run(nll_loss_op)
    temperature = sess.run(temp_var)
    acc = sess.run(acc_op)

    expected_ece = sess.run(ece)

    labels = sess.run(labels_tensor)
    before_logits = sess.run(logits_tensor)
    before_fscore = precision_recall_fscore_support(labels, np.argmax(before_logits, axis=1), average='binary')

    after_logits = sess.run(logits_w_temp)
    after_softmax = softmax(after_logits)
    index = np.argmax(after_softmax, axis=1)
    probs = np.amax(after_softmax, axis=1)

    after_fscore = precision_recall_fscore_support(labels, np.argmax(after_softmax, axis=1), average='binary')
    print(np.argmax(softmax(after_logits), axis=1))
    print(labels)

    before_softmax = softmax(before_logits)
    probss = np.amax(before_softmax, axis=1)

    print("Optimal temperature: {:.2f}".format(temperature[0]))
    print("Confidence (before): {:.3f}".format(np.mean(probss)))
    print("Confidence (after): {:.3f}".format(np.mean(probs)))
    print("Precision: {:.3f},  Recall: {:.3f},  F-score: {:.3f}".format(after_fscore[0], after_fscore[1], after_fscore[2]))

    after_ece = tfp.stats.expected_calibration_error(10, logits_w_temp, labels_tensor, name='ece')
    after_expected_ece = sess.run(after_ece)
    print ("Before Calibration： NLL: {:.3f}, ECE: {:.3f}".format(org_nll_loss, expected_ece * 100))
    print ("After Calibration： NLL: {:.3f}, ECE: {:.3f}".format(nll_loss, after_expected_ece * 100))

    return temp_var

def softmax(x):
    s = np.max(x, axis=1)
    s = s[:, np.newaxis]
    e_x = np.exp(x-s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]

    return e_x / div

def preprocess_test():
    print("Loading data...")
    x_text, y = data_helpers.load_data_and_labels("./data/cris/test.pos", "./data/cris/test.neg")

    # Build vocabulary
#    max_document_length = 8192#max([len(x.split(" ")) for x in x_text])
#    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    return x_shuffled, y_shuffled

def main(args):
  FLAGS = tf.flags.FLAGS
  x_train, x_test, y_train, y_test = train_test_split(x_shuffled, y_shuffled, test_size=0.2, random_state=42, stratify=y_shuffled)

  print("\nEvaluating...\n")
    # Evaluation
    # ==================================================
  checkpoint_file = tf.train.latest_checkpoint(args.checkpoint_dir)
  graph = tf.Graph()
  with graph.as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True,
                                  log_device_placement=False,
                                  gpu_options=tf.GPUOptions(force_gpu_compatible=True, allow_growth=True)
                                 )
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        input_y = graph.get_operation_by_name("input_y").outputs[0]

        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        scores = graph.get_operation_by_name("output/scores").outputs[0]
        batches = data_helpers.batch_iter(list(x_test), 16, 1, shuffle=False)

        all_predictions, all_logits = [], []
        for x_test_batch in batches:
            batch_predictions, logits = sess.run([predictions, scores], {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])
            all_logits.append(logits)
        all_logits = np.concatenate(all_logits)

        print ("Logits get! Do temperature scaling...")
        print ("=" * 80)
        temp_var = temp_scaling(all_logits, y_test, sess)
        print ("=" * 80)
        print ("Done!")

if __name__ == '__main__':
    main(args)
