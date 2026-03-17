import tensorflow as tf
from tensorflow.keras import layers, models

def create_rnn_model(vocab_size, embedding_dim, rnn_units, num_classes):
    model = models.Sequential([
        layers.Embedding(vocab_size, embedding_dim, mask_zero=True),
        layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(rnn_units)),
        layers.Dense(64, activation=\'relu\'),
        layers.Dense(num_classes, activation=\'softmax\')
    ])
    return model

def train_rnn_model(model, train_dataset, test_dataset, epochs=10):
    model.compile(optimizer=\'adam\', loss=\'sparse_categorical_crossentropy\', metrics=[\\'accuracy\\'])
    history = model.fit(train_dataset, epochs=epochs, validation_data=test_dataset)
    return history

if __name__ == \'__main__\':
    # Load and preprocess the IMDB dataset
    vocab_size = 10000
    (train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.imdb.load_data(num_words=vocab_size)

    # Pad sequences to a fixed length
    max_len = 256
    train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data, maxlen=max_len)
    test_data = tf.keras.preprocessing.sequence.pad_sequences(test_data, maxlen=max_len)

    # Create TensorFlow datasets
    BUFFER_SIZE = 10000
    BATCH_SIZE = 64
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels))
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Model parameters
    embedding_dim = 128
    rnn_units = 64
    num_classes = 2  # Sentiment: positive or negative

    # Create and train the RNN model
    rnn_model = create_rnn_model(vocab_size, embedding_dim, rnn_units, num_classes)
    rnn_model.summary()
    train_rnn_model(rnn_model, train_dataset, test_dataset, epochs=5)

    loss, accuracy = rnn_model.evaluate(test_dataset, verbose=2)
    print(f"\nTest accuracy: {accuracy}")
