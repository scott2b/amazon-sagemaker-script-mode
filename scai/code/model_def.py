import tensorflow as tf
#from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import tensorflow_hub as hub
import tensorflow_text
from tensorflow.keras.optimizers import Adam, SGD, RMSprop


#HEIGHT = 32
#WIDTH = 32
#DEPTH = 3
#NUM_CLASSES = 10

# hyperparameters
#BATCH_SIZE = 16 # seems to significantly affect the validation curves
#SEED = 123 # random seed
#EPOCHS = 20 # Max epochos. Early stopping is
#EARLY_STOPPING_PATIENCE = 0


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


tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1'
#tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/albert_en_preprocess/2'
#tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/albert_en_base/2'


def build_classifier_model():
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
  preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
  #preprocessing_layer = make_preprocess_model(['text'], tfhub_handle_preprocess)
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
    if optimizer.lower() == 'sgd':
        opt = SGD(lr=learning_rate * size, decay=weight_decay, momentum=momentum)
    elif optimizer.lower() == 'rmsprop':
        opt = RMSprop(lr=learning_rate * size, decay=weight_decay)
    else:
        opt = Adam(lr=learning_rate * size, decay=weight_decay)
    classifier_model = build_classifier_model()
    classifier_model.compile(
        optimizer=opt,
        loss=get_loss_function(),
        metrics=get_metrics_function())
        #loss='binary_crossentropy',
        #metrics=['accuracy'])
    return classifier_model
