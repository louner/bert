from run_classifier import *

data_dir = 'data/MRPC/MRPC/'
output_dir = 'output/'
vocab_file = 'model/uncased_L-12_H-768_A-12/vocab.txt'
do_lower_case = True
batch_size = 16
drop_remainder = True
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


max_seq_length = 128
processor = MrpcProcessor()
label_list = processor.get_labels()
train_examples = processor.get_train_examples(data_dir)
train_file = os.path.join(output_dir, "train.tf_record")
file_based_convert_examples_to_features(train_examples, label_list, max_seq_length, tokenizer, train_file)
d = tf.data.TFRecordDataset(train_file)
d = d.apply(tf.contrib.data.map_and_batch(lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))
