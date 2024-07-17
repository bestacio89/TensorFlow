
# Data Package README

The `data/` package in this repository includes `datasets.py`, where functions for data loading and preprocessing are defined. This README outlines how to utilize and customize these functions to handle data effectively in your TensorFlow-based deep learning projects.

## `datasets.py` - Data Handling and Preprocessing

### Example Data Functions

#### 1. `load_data()`

An example function to load and preprocess image data using TensorFlow's `ImageDataGenerator`:

```python
import tensorflow as tf

def load_data(data_dir, img_height=256, img_width=256, batch_size=32):
    """
    Loads and preprocesses image data from a directory using TensorFlow's ImageDataGenerator.

    Args:
    - data_dir (str): Directory path containing subdirectories of images
    - img_height (int): Height to which images will be resized
    - img_width (int): Width to which images will be resized
    - batch_size (int): Batch size for training

    Returns:
    - train_generator (tf.keras.preprocessing.image.DirectoryIterator): Generator for training data
    - validation_generator (tf.keras.preprocessing.image.DirectoryIterator): Generator for validation data
    """
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    return train_generator, validation_generator
```

#### 2. `preprocess_data()`

Function to preprocess data, such as normalization or feature scaling:

```python
def preprocess_data(data):
    """
    Preprocesses the input data.

    Args:
    - data (numpy.ndarray or pandas.DataFrame): Input data to be preprocessed

    Returns:
    - preprocessed_data (numpy.ndarray): Preprocessed data
    """
    # Example: Normalize data
    preprocessed_data = data / 255.0  # Normalize pixel values to [0, 1]
    
    # Example: Perform additional preprocessing steps if needed
    
    return preprocessed_data
```

### Usage Example

#### Example Script (`example_script.py`)

An example script demonstrating how to use `datasets.py` functions to load and preprocess data:

```python
from data.datasets import load_data, preprocess_data
from models.model import build_simple_cnn
from utils.helpers import plot_history

# Example: Loading and preprocessing image data
data_dir = 'data/images/'
train_generator, validation_generator = load_data(data_dir)

# Example: Preprocessing numerical data
# Assuming `X_train` is your numerical data
X_train_preprocessed = preprocess_data(X_train)

# Example: Building and training a CNN
input_shape = train_generator.image_shape
num_classes = len(train_generator.class_indices)
model = build_simple_cnn(input_shape, num_classes)

history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Example: Plotting training history
plot_history(history)
```

### Conclusion

The `datasets.py` file in `data/` provides functions to handle data loading and preprocessing, crucial for deep learning projects using TensorFlow. Customize these functions or add new ones as needed to accommodate different data formats and preprocessing requirements. This README serves as a guide to effectively integrate and utilize data handling functions in your deep learning workflows.
