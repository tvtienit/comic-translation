import tensorflow as tf
from tensorflow.python.lib.io import file_io
import urllib.request
import collections
import math
import os
import random
import zipfile
import datetime as dt
import numpy as np

from embedding import constant

flags = tf.app.flags
FLAGS = flags.FLAGS

def set_train_flags():
  global flags
  flag_keys = ['output_dir', 'input_dir', 'num_steps', 'gs', 'vocabulary_size', 'batch_size', 'skip_window', 'num_skips',
               'embedding_size', 'language', 'voc_data']
  flag_types = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  flag_default_values = [constant.LOCAL_OUTPUT, constant.LOCAL_INPUT, constant.DEFAULT_NUM_STEP,
                         constant.DEFAULT_GS_ENV, constant.VOCABULARY_SIZE, constant.BATCH_SIZE,
                         constant.SKIP_WINDOW, constant.NUM_SKIPS, constant.EMBEDDING_SIZE, constant.LANGUAGE, constant.VOCABULARY_DATA]
  flag_default_description = 'Understand it urself'
  for i in range(len(flag_keys)):
    print('flag ' + flag_keys[i])
    if flag_types[i] == 0:
      flags.DEFINE_string(flag_keys[i], flag_default_values[i], flag_default_description)
    else:
      flags.DEFINE_integer(flag_keys[i], flag_default_values[i], flag_default_description)

  print('[INFO] Configure flag successfully')

set_train_flags()
vocabulary_size = FLAGS.vocabulary_size

def maybe_download(filename, url, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename

def read_file(filename, method, encoding):
  encoding_type = encoding
  if encoding is None:
    encoding_type = 'ascii'
  if method == 1:
    with tf.gfile.GFile(filename, "r") as g_file:
      return g_file
  with open(filename, 'r', encoding = encoding_type) as lcl_file:
    return lcl_file

# Read the data into a list of strings.
def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    file_io = read_file(filename, FLAGS.gs, None)
    data = tf.compat.as_str(file_io.read()).split()
    return data

def build_dataset(words, n_words):
    global vocabulary_size
    """Process raw inputs into a dataset."""
    count = []
    filename = os.path.join(FLAGS.voc_data, 'train_' + FLAGS.language + '_vocabulary.txt')
    print("VOCA_PATH: " + filename)
    print("FILE: " + filename)
    encoding_type = 'utf-8'
    #count.extend(collections.Counter(words).most_common(n_words - 1))
    file_io = read_file(filename, FLAGS.gs, encoding_type)
    for line in file_io:
      count.append([line[:-1],1])
    print(count[:5])
    vocabulary_size=len(count)
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()

    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


def collect_data(vocabulary_size=10000):
  #  url = 'http://mattmahoney.net/dc/'
   # filename = maybe_download('text8.zip', url, 31344016)
    input_path = os.path.join(FLAGS.input_dir, "train." + FLAGS.language)
    print("INPUT PATH: " + input_path)
    vocabulary = read_data(input_path)
    print(vocabulary[:7])
    data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                                vocabulary_size)
    del vocabulary  # Hint to reduce memory.
    return data, count, dictionary, reverse_dictionary

data_index = 0
# generate batch data
def generate_batch(data, batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    context = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window input_word skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # input word at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]  # this is the input word
            context[i * num_skips + j, 0] = buffer[target]  # these are the context words
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, context

data, count, dictionary, reverse_dictionary = collect_data(vocabulary_size=vocabulary_size)

batch_size = 128
embedding_size = 56  # Dimension of the embedding vector.
skip_window = 4       # How many words to consider left and right.
num_skips = 2      # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default():

  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_context = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Look up embeddings for inputs.
  embeddings = tf.Variable(
      tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
  embed = tf.nn.embedding_lookup(embeddings, train_inputs)

  # Construct the variables for the softmax
  weights = tf.Variable(
      tf.truncated_normal([embedding_size, vocabulary_size],
                          stddev=1.0 / math.sqrt(embedding_size)))
  biases = tf.Variable(tf.zeros([vocabulary_size]))
  hidden_out = tf.transpose(tf.matmul(tf.transpose(weights), tf.transpose(embed))) + biases

  # convert train_context to a one-hot format
  train_one_hot = tf.one_hot(train_context, vocabulary_size)

  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hidden_out, labels=train_one_hot))

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(cross_entropy)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)

  # Add variable initializer.
  init = tf.global_variables_initializer()

def get_file_storage(output_path):
  if FLAGS.gs == 1:
    return file_io.FileIO(output_path, 'w')
  return output_path


def run(graph, num_steps):
    print("START WITHIN " + str(num_steps) + " STEPS")
    with tf.Session(graph=graph) as session:
      # We must initialize all variables before we use them.
      init.run()
      print('Initialized')

      average_loss = 0
      for step in range(num_steps):
        batch_inputs, batch_context = generate_batch(data,
            batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_context: batch_context}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = session.run([optimizer, cross_entropy], feed_dict=feed_dict)
        average_loss += loss_val
    
        if step % 2000 == 0:
          if step > 0:
            average_loss /= 2000
          # The average loss is an estimate of the loss over the last 2000 batches.
          print('Average loss at step ', step, ': ', average_loss)
          average_loss = 0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
          sim = similarity.eval()
          for i in range(valid_size):
            valid_word = reverse_dictionary[valid_examples[i]]
            top_k = 8  # number of nearest neighbors
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % valid_word
            for k in range(top_k):
              close_word = reverse_dictionary[nearest[k]]
              log_str = '%s %s,' % (log_str, close_word)
            print(log_str)
      final_embeddings = normalized_embeddings.eval()
      return final_embeddings
      #np.save(output_path, np.array(final_embeddings))

num_steps = 100000
softmax_start_time = dt.datetime.now()
final_embeddings= run(graph, num_steps=num_steps)
output_path = get_file_storage(os.path.join(FLAGS.output_dir, "word_embeding_" + FLAGS.language + ".npy"))
np.save(output_path, np.array(final_embeddings))

softmax_end_time = dt.datetime.now()

print("Softmax method took {} minutes to run 100 iterations".format((softmax_end_time-softmax_start_time).total_seconds()))
