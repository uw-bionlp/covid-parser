wget -O pretrained_bert_tf.tar.gz https://www.dropbox.com/s/8armk04fu16algz/pretrained_bert_tf.tar.gz?dl=1 -q --show-progress --progress=bar:force 2>&1 \
tar -xzf pretrained_bert_tf.tar.gz                                                                      \
tar -xzf pretrained_bert_tf/biobert_pretrain_output_all_notes_150000.tar.gz                             \
mv biobert_pretrain_output_all_notes_150000 pretrained/                                                 \
cd pretrained/biobert_pretrain_output_all_notes_150000/ && mv bert_config.json config.json && cd ../..  \
rm -rf pretrained_bert_tf                                                                               \
rm pretrained_bert_tf.tar.gz