from tensorflow.keras import layers, Model
import tensorflow_hub as hub

def build_model(train_dataset):
    """
    Trains deep learning model for finance complaint classifier.
    """

    # Define input layer
    inputs = layers.Input(shape=[], dtype='string')

    # Download pretrained embedding layer from tensorflow hub
    print("Downloading pretrained layers from tensorflow hub...")
    hub_embedding_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                                        trainable=False,
                                        name="universal_sentence_encoder")
    print("Successfully downloaded pretrained layers...")

    # Define embedding layer of model and pass inputs
    embedded_inputs = hub_embedding_layer(inputs)

    # Define hidden layers
    x = layers.Dense(128, activation='relu')(embedded_inputs) # pass inputs from embedding layer
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)

    # Define output layer
    outputs = layers.Dense(12, # number of classes
                           activation='softmax')(x)
    
    # Build model
    clf = Model(inputs, outputs, name="clf")

    # Compile model
    clf.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    
    # Train model
    clf.fit(train_dataset,
            steps_per_epoch=int(0.1*len(train_dataset)),
            epochs=10,
            verbose=0)

    return clf

def save_model(model, model_filepath):
    """
    Save model to hdf5 format.
    """
    model.save(f'{model_filepath}/classifier.h5')

def evaluate_model(model, test_data):
    loss, accuracy = model.evaluate(test_data)
    print(f"The model\'s loss on the test data is {loss: .2f}.")
    print(f"The model\'s accuracy on the test data is {accuracy: .2f}.")

