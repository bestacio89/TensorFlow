
# Utils Package README

The `utils/` package in this repository contains helper functions (`helpers.py`) designed to assist various tasks in your TensorFlow-based deep learning project. This README provides an overview of the functions included and how to use them effectively.

## `helpers.py` Functions

### Example Functions

#### 1. `plot_history()`

This function plots training history such as accuracy and loss over epochs using Matplotlib.

```python
import matplotlib.pyplot as plt

def plot_history(history):
    """
    Plots accuracy and loss curves for training and validation.
    
    Args:
    - history (History object): TensorFlow History object returned from model.fit()
    """
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()

    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()
```

#### 2. `save_model()` and `load_model()`

Functions to save and load trained models using TensorFlow's `model.save()` and `tf.keras.models.load_model()`.

```python
from tensorflow.keras.models import load_model

def save_model(model, filepath):
    """
    Saves the trained model to a specified filepath.
    
    Args:
    - model (tf.keras.Model): Trained Keras model object
    - filepath (str): Path to save the model
    """
    model.save(filepath)

def load_model(filepath):
    """
    Loads a pre-trained model from a specified filepath.
    
    Args:
    - filepath (str): Path from which to load the model
    
    Returns:
    - model (tf.keras.Model): Loaded Keras model object
    """
    model = load_model(filepath)
    return model
```

### Usage Example

#### Example Script (`example_script.py`)

An example script demonstrating how to use `helpers.py` functions:

```python
from utils.helpers import plot_history, save_model, load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Example: Training a simple neural network
model = Sequential([
    Dense(64, activation='relu', input_shape=(784,)),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

# Plot training history
plot_history(history)

# Save trained model
save_model(model, 'saved_models/my_model')

# Load model and evaluate
loaded_model = load_model('saved_models/my_model')
evaluation = loaded_model.evaluate(x_test, y_test)
print("Evaluation Metrics:", evaluation)
```

### Conclusion

The `helpers.py` file in `utils/` provides essential functions for saving/loading models and plotting training history. These utilities streamline common tasks in deep learning projects, enhancing productivity and code organization. Customize these functions or add new ones as needed to suit your project requirements.
