import tensorflow as tf
import sys

fname = sys.argv[1]
for i,example in enumerate(tf.python_io.tf_record_iterator(fname)):
    result = tf.train.Example.FromString(example)
    print result
    if i > 10:
        break

