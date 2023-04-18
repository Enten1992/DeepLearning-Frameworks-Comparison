
import tensorflow as tf
from tensorflow.keras import layers, models

def build_rnn_model(vocab_size, embedding_dim, rnn_units, batch_size):
    """
    Builds a recurrent neural network (RNN) model using TensorFlow/Keras for text generation.

    Args:
        vocab_size (int): The size of the vocabulary.
        embedding_dim (int): The dimension of the embedding layer.
        rnn_units (int): The number of RNN units.
        batch_size (int): The batch size for training.

    Returns:
        tf.keras.Model: The compiled RNN model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer=\'glorot_uniform\'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

def train_rnn_model(model, dataset, epochs=10):
    """
    Trains the given RNN model.

    Args:
        model (tf.keras.Model): The compiled RNN model to train.
        dataset (tf.data.Dataset): The training dataset.
        epochs (int): Number of epochs to train for.

    Returns:
        tf.keras.callbacks.History: Training history object.
    """
    print("\nStarting RNN model training...")
    history = model.fit(dataset, epochs=epochs)
    print("RNN model training finished.")
    return history

def generate_text(model, start_string, num_generate=100, temperature=1.0):
    """
    Generates text using the trained RNN model.

    Args:
        model (tf.keras.Model): The trained RNN model.
        start_string (str): The starting string for text generation.
        num_generate (int): The number of characters to generate.
        temperature (float): Controls the randomness of predictions.

    Returns:
        str: The generated text.
    """
    # Evaluation step (the `build_model` method might be different here)
    # We need to rebuild the model with a batch_size=1
    # For simplicity, we\'ll assume the model can handle variable batch sizes for generation
    # In a real scenario, you\'d typically save and reload the weights into a new model with batch_size=1

    input_eval = [char_to_int[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)

        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_outputs=1)[-1, 0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(int_to_char[predicted_id])

    return start_string + \'\'.join(text_generated)

if __name__ == "__main__":
    # Example usage with dummy data (replace with actual text dataset)
    print("Running example RNN model with dummy data.")
    
    # Dummy text data
    text = "selamun aleykum nasilsin iyimisin" * 100
    
    # Create vocabulary
    vocab = sorted(set(text))
    char_to_int = {c: i for i, c in enumerate(vocab)}
    int_to_char = {i: c for i, c in enumerate(vocab)}
    
    # Convert text to integers
    text_as_int = [char_to_int[c] for c in text]

    # Create training examples and targets
    seq_length = 100
    examples_per_epoch = len(text) // (seq_length + 1)

    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

    sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    # Model parameters
    VOCAB_SIZE = len(vocab)
    EMBEDDING_DIM = 256
    RNN_UNITS = 1024

    rnn_model = build_rnn_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
    rnn_model.summary()

    # Define loss function
    def loss(labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

    rnn_model.compile(optimizer=\'adam\', loss=loss)

    train_rnn_model(rnn_model, dataset, epochs=1)

    # Generate text (rebuild model with batch_size=1 for generation)
    # This is a simplified approach; in a real scenario, you\'d save and load weights.
    # For demonstration, we\'ll use the trained model directly, assuming it can handle it.
    # A more robust approach would be to create a new model with batch_size=1 and load weights.
    # For this example, we\'ll just show the generation logic.
    
    # Rebuild the model with batch_size=1 for text generation
    inference_model = build_rnn_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, batch_size=1)
    inference_model.set_weights(rnn_model.get_weights())
    
    print("\nGenerated Text:")
    print(generate_text(inference_model, start_string="selamun", num_generate=200))
