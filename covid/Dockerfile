FROM python:3.7 as build

WORKDIR /usr/src/app

# Copy code
COPY grpc_server /usr/src/app/grpc_server
COPY annotate    /usr/src/app/annotate
COPY corpus      /usr/src/app/corpus
COPY models      /usr/src/app/models
COPY utils       /usr/src/app/utils
COPY main.py process.py constants.py requirements.txt /usr/src/app/

# Install wget and other packages
RUN apt-get update \
    && apt-get install -y wget gcc libc-dev g++ libffi-dev libxml2 libffi-dev -y \
    && rm -rf /var/lib/apt/lists/* 

# Load clinicalBERT
RUN mkdir pretrained
RUN wget -O pretrained_bert_tf.tar.gz https://www.dropbox.com/s/8armk04fu16algz/pretrained_bert_tf.tar.gz?dl=1 -q --show-progress --progress=bar:force 2>&1 \
    && tar -xzf pretrained_bert_tf.tar.gz                                                                      \
    && tar -xzf pretrained_bert_tf/biobert_pretrain_output_all_notes_150000.tar.gz                             \
    && mv biobert_pretrain_output_all_notes_150000 pretrained/                                                 \
    && cd pretrained/biobert_pretrain_output_all_notes_150000/ && mv bert_config.json config.json && cd ../..  \
    && rm -rf pretrained_bert_tf                                                                               \
    && rm pretrained_bert_tf.tar.gz

# pip install
RUN python3 -m pip install --no-cache-dir -r requirements.txt

CMD [ "python", "./main.py" ]