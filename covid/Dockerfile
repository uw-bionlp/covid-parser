FROM python:3-slim as build

WORKDIR /usr/src/app

# Copy code
COPY grpc_server /usr/src/app/grpc_server
COPY annotate    /usr/src/app/annotate
COPY corpus      /usr/src/app/corpus
COPY models      /usr/src/app/models
COPY utils       /usr/src/app/utils
COPY main.py process.py constants.py requirements.txt /usr/src/app/

# Load clinicalBERT
RUN wget -O pretrained_bert_tf.tar.gz https://www.dropbox.com/s/8armk04fu16algz/pretrained_bert_tf.tar.gz?dl=1 && \
    tar -xzf pretrained_bert_tf.tar.gz                                                                         && \
    tar -xzf pretrained_bert_tf/biobert_pretrain_output_all_notes_150000.tar.gz                                && \
    mkdir pretrained                                                                                           && \ 
    mv biobert_pretrain_output_all_notes_150000 pretrained/                                                    && \
    rm -rf pretrained_bert_tf

# pip install
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

FROM gcr.io/distroless/python3
COPY --from=build /usr/src/app /app
WORKDIR /app

CMD [ "main.py" ]