
import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_model(input_shape=(32, 32, 3), num_classes=10):
    """
    Builds a convolutional neural network (CNN) model using TensorFlow/Keras.

    Args:
        input_shape (tuple): The shape of the input images (height, width, channels).
        num_classes (int): The number of output classes for classification.

    Returns:
        tf.keras.Model: The compiled CNN model.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation=\'relu\', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation=\'relu\'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation=\'relu\'),
        layers.Flatten(),
        layers.Dense(64, activation=\'relu\'),
        layers.Dense(num_classes, activation=\'softmax\')
    ])

    model.compile(optimizer=\'adam\', 
                  loss=\'sparse_categorical_crossentropy\', 
                  metrics=[\'accuracy\'])
    return model

def train_cnn_model(model, train_images, train_labels, epochs=10, batch_size=32):
    """
    Trains the given CNN model.

    Args:
        model (tf.keras.Model): The compiled CNN model to train.
        train_images (np.array): Training image data.
        train_labels (np.array): Training labels.
        epochs (int): Number of epochs to train for.
        batch_size (int): Batch size for training.

    Returns:
        tf.keras.callbacks.History: Training history object.
    """
    print("\nStarting CNN model training...")
    history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size)
    print("CNN model training finished.")
    return history

def evaluate_cnn_model(model, test_images, test_labels):
    """
    Evaluates the given CNN model.

    Args:
        model (tf.keras.Model): The trained CNN model to evaluate.
        test_images (np.array): Test image data.
        test_labels (np.array): Test labels.

    Returns:
        tuple: Loss and accuracy of the model on the test set.
    """
    print("\nEvaluating CNN model...")
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    return test_loss, test_acc

if __name__ == "__main__":
    # Example usage with dummy data (replace with actual dataset like CIFAR-10)
    print("Running example CNN model with dummy data.")
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Select a subset for quicker demonstration
    train_images = train_images[:1000]
    train_labels = train_labels[:1000]
    test_images = test_images[:200]
    test_labels = test_labels[:200]

    cnn_model = build_cnn_model()
    cnn_model.summary()

    train_cnn_model(cnn_model, train_images, train_labels, epochs=1)
    evaluate_cnn_model(cnn_model, test_images, test_labels)
