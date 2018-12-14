export bert_base_dir=model/uncased_L-12_H-768_A-12
export glue_dir=data/quora/toy
#export glue_dir=data/quora/toy
export trained_classifier=/tmp/mrpc_sentence_difference

python create_pretraining_data.py \
  --input_file=$glue_dir/question_sentences.txt \
  --output_file=/tmp/tf_examples.tfrecord \
  --vocab_file=$bert_base_dir/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  -tion_sentences.txtrandom_seed=12345 \
  --dupe_factor=5

rm -rf /tmp/pretraining_output/
train_size=`python read_tfrecord.py /tmp/tf_examples.tfrecord | grep input_ids | wc -l`
train_batch_size=10
epoch_batch_num=$((train_size/train_batch_size))
train_epoch_num=3

python run_pretraining.py \
  --input_file=/tmp/tf_examples.tfrecord \
  --output_dir=/tmp/pretraining_output \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$bert_base_dir/bert_config.json \
  --init_checkpoint=$bert_base_dir/bert_model.ckpt \
  --train_batch_size=$train_batch_size \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=$((epoch_batch_num*train_epoch_num)) \
  --num_warmup_steps=10 \
  --learning_rate=2e-5 \
  --eval_batch_size=$train_batch_size \
  --max_eval_steps=$epoch_batch_num
