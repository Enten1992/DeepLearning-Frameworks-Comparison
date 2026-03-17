import tensorflow as tf
from tensorflow.keras import layers, models

def create_cnn_model(input_shape, num_classes):
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
    return model

def train_cnn_model(model, train_images, train_labels, test_images, test_labels, epochs=10, batch_size=32):
    model.compile(optimizer=\'adam\', loss=\'sparse_categorical_crossentropy\', metrics=[\'accuracy\'])
    history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, 
                        validation_data=(test_images, test_labels))
    return history

if __name__ == \'__main__\':
    # Load and preprocess the MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    # Reshape images to include channel dimension (for grayscale, it's 1)
    train_images = train_images.reshape((60000, 28, 28, 1)).astype(\'float32\') / 255
    test_images = test_images.reshape((10000, 28, 28, 1)).astype(\'float32\') / 255

    # Get input shape and number of classes
    input_shape = train_images.shape[1:]
    num_classes = len(tf.unique(train_labels)[0])

    # Create and train the CNN model
    cnn_model = create_cnn_model(input_shape, num_classes)
    print(\
    cnn_model.summary()
    train_cnn_model(cnn_model, train_images, train_labels, test_images, test_labels, epochs=5)
    
    loss, accuracy = cnn_model.evaluate(test_images, test_labels, verbose=2)
    print(f"\nTest accuracy: {accuracy}")
