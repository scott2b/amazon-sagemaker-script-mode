import tensorflow as tf
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

import datetime
import os
import shutil
import time
import tensorflow_hub as hub
import tensorflow_text as text
import matplotlib.pyplot as plt
from psutil import virtual_memory
from pathlib import Path
from official.nlp import optimization  # to create AdamW optmizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


HEIGHT = 32
WIDTH = 32
DEPTH = 3
NUM_CLASSES = 10

# hyperparameters
BATCH_SIZE = 16 # seems to significantly affect the validation curves
SEED = 123 # random seed
EPOCHS = 20 # Max epochos. Early stopping is
EARLY_STOPPING_PATIENCE = 0


def make_preprocess_model(sentence_features, preprocessor_url):
    """This is mostly used so that we can utilize the full 512 sequence
    length limitation, as the Keras hub preprocessors will default to
    128 tokens. There is probably an easier way to do this?
    """
    input_segments = [
        tf.keras.layers.Input(shape=(), dtype=tf.string, name=ft)
        for ft in sentence_features]
    # Tokenize the text to word pieces.
    bert_preprocess = hub.load(preprocessor_url)
    tokenizer = hub.KerasLayer(bert_preprocess.tokenize, name='tokenizer')
    segments = [tokenizer(s) for s in input_segments]
    # Optional: Trim segments in a smart way to fit seq_length.
    # Simple cases (like this example) can skip this step and let
    # the next step apply a default truncation to approximately equal lengths.
    truncated_segments = segments
    # Pack inputs. The details (start/end token ids, dict of output tensors)
    # are model-dependent, so this gets loaded from the SavedModel.
    packer = hub.KerasLayer(bert_preprocess.bert_pack_inputs,
                            arguments=dict(seq_length=512),
                            name='packer')
    model_inputs = packer(truncated_segments)
    return tf.keras.Model(input_segments, model_inputs)


def build_classifier_model():
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
  #preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
  preprocessing_layer = make_preprocess_model(['text'], tfhub_handle_preprocess)
  encoder_inputs = preprocessing_layer(text_input)
  encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
  outputs = encoder(encoder_inputs)
  net = outputs['pooled_output']
  net = tf.keras.layers.Dropout(0.1)(net)
  # TODO: It seems like we could output a sigmoid activation instead of having to apply sigmoid after the fact?
  net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
  return tf.keras.Model(text_input, net)


def get_loss_function():
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)

def get_metrics_function():
    return tf.metrics.BinaryAccuracy()


def get_model(learning_rate, weight_decay, optimizer, momentum, size, mpi=False, hvd=False):

    #steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    #num_train_steps = steps_per_epoch * EPOCHS
    #num_warmup_steps = int(0.1*num_train_steps)
    #init_lr = 5e-5
    #optimizer = optimization.create_optimizer(
    #    init_lr=init_lr,
    #    num_train_steps=num_train_steps,
    #    num_warmup_steps=num_warmup_steps,
    #    optimizer_type='adamw')
    #init_lr = 1.1999607522739098e-06
    #optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    classifier_model = build_classifier_model()
    earlystop_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=EARLY_STOPPING_PATIENCE)
    classifier_model.compile(
        optimizer=optimizer,
        loss=get_loss_function(),
        metrics=get_metrics_function())
    #print(f'Training model with {tfhub_handle_encoder}')
    #start = time.time()
    #history = classifier_model.fit(
    #    x=train_ds,
    #    validation_data=val_ds,
    #    callbacks=[earlystop_callback],
    #    epochs=EPOCHS)
    #end = time.time()
    #print('Completed training in (minutes):', (end - start)/60)
    return classifier_model
