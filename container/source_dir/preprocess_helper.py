import os
import time
from html.parser import HTMLParser

import boto3
import nltk
import numpy as np
import tensorflow_hub as hub
from html2text import html2text
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from numpy import genfromtxt
import tarfile

USE_MODEL_TAR_GZ = 'universal-sentence-encoder_4.tar.gz'
USE_MODEL_DIR = 'universal-sentence-encoder_4'

porter_stemmer = PorterStemmer()
nltk.download('stopwords')
stopwords_list = stopwords.words('english')
html_parse = HTMLParser()
mail_body_bucket_name = 'dev.mail.bodies'
sagemaker_bucket = 'sagemaker-neural-net'


# def load_universal_sentence_encoder():
#     sentence_encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
#     return sentence_encoder


def filter_token(token):
    if token.isnumeric() or len(token) <= 2 or len(token) >= 26 or token in stopwords_list:
        return False
    return True


def remove_stop_words(tokens):
    filtered_tokens = []
    for token in tokens:
        if token not in stopwords_list:
            filtered_tokens.append(token)
    return filtered_tokens


def perform_stemming(tokens):
    stemmed_tokens = []
    for token in tokens:
        stemmed_tokens.append(porter_stemmer.stem(token))
    return stemmed_tokens


def extract_text_from_html(body):
    return html2text(body)


def pre_process_mail_body(body):
    body = extract_text_from_html(body)
    body = body.lower()
    tokens = word_tokenize(body)
    tokens = filter(filter_token, tokens)
    return perform_stemming(tokens)


def pre_process_mails(mail_bodies_list):
    processed_mail_bodies = []
    for mail_body in mail_bodies_list:
        processed_mail_bodies.append(pre_process_mails(mail_body))
    return processed_mail_bodies


def download_mail_bodies_from_s3(data_dir):
    s3_client = boto3.client('s3')
    mail_body_keys = genfromtxt(os.path.join(data_dir, 'mids.csv'), delimiter=",", dtype=str)
    mail_bodies = []
    mail_bodies_labels = []
    for mail_body_key in mail_body_keys:
        s3_client.download_file(mail_body_bucket_name, mail_body_key, mail_body_key)
        fp = open(mail_body_key, "r")
        mail_bodies.append(fp.read())
        mail_bodies_labels.append([1 if mail_body_key.startswith('ham') else 0])
    return mail_bodies, mail_bodies_labels


def download_univ_sent_enc_model_from_s3():
    start = time.time()
    print('Started downloading USE from s3')
    s3_client = boto3.client('s3')
    s3_client.download_file(sagemaker_bucket, USE_MODEL_TAR_GZ, USE_MODEL_TAR_GZ)
    tar = tarfile.open(USE_MODEL_TAR_GZ)
    tar.extractall(os.path.join(os.getcwd(), USE_MODEL_DIR))
    tar.close()
    print('Completed downloading USE from s3')
    end = time.time()
    print(f"Time taken : {end - start}")


def create_training_data(data_dir):
    download_univ_sent_enc_model_from_s3()
    sentence_encoder = hub.load(os.path.join(os.getcwd(), USE_MODEL_DIR))
    print(sentence_encoder(['hello']))
    mail_bodies, mail_bodies_labels = download_mail_bodies_from_s3(data_dir)
    train_vectors = [sentence_encoder([body]) for body in mail_bodies]
    train_vectors = np.array(train_vectors)
    train_vectors = np.reshape(train_vectors, (train_vectors.shape[0], train_vectors.shape[2]))
    return [np.array(train_vectors), np.array(mail_bodies_labels)]
