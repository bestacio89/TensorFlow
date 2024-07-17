# Config Package README

The `config/` package in this repository contains various modules that centralize configuration settings for different aspects of your TensorFlow-based deep learning project. This README outlines the structure of the `config/` package and demonstrates how to utilize its contents for configuring models, training processes, and other project settings.

## Structure of `config/`

The `config/` package is structured as follows:

```
config/
│
├── __init__.py          # Initialize the config package
├── hyperparameters.py   # Hyperparameter settings
├── paths.py             # File paths configuration
├── model_config.py      # Model architecture configurations
└── training_config.py   # Training settings and configurations
```

### Modules Description

- **`__init__.py`**: Initializes the `config` package, allowing modules within it to be imported.
  
- **`hyperparameters.py`**: Defines hyperparameters such as learning rates, batch sizes, and epochs used for training models.

- **`paths.py`**: Stores file paths for data, saved models, logs, and any other necessary files or directories.

- **`model_config.py`**: Contains configurations related to model architecture, including layer definitions, activation functions, and other architectural choices.

- **`training_config.py`**: Specifies training settings and configurations such as optimizer choices, loss functions, metrics, and callbacks used during model training.

## Usage Example

### Example Script (`train.py`)

An example script demonstrating how to import and use configurations from the `config/` package in your training process:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from config.hyperparameters import batch_size, epochs
from config.paths import data_dir, saved_models_dir, logs_dir
from config.model_config import model_layers
from config.training_config import optimizer, loss_function, metrics
from data.datasets import load_data

# Example: Load and preprocess data
train_generator, validation_generator = load_data(data_dir)

# Example: Build model using configurations from model_config.py
model = Sequential(model_layers)
model.compile(optimizer=optimizer,
              loss=loss_function,
              metrics=metrics)

# Example: Train model
history = model.fit(
    train_generator,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=validation_generator
)

# Example: Save trained model
model.save(saved_models_dir + 'my_model')

# Example: Logging
with open(logs_dir + 'training.log', 'a') as f:
    f.write('Training completed.\n')

print("Training completed successfully.")
```

### Customization

- **Modify Configurations**: Update values in `hyperparameters.py`, `paths.py`, `model_config.py`, and `training_config.py` according to your project requirements.
  
- **Extend Functionality**: Add new configuration modules or update existing ones to accommodate additional project settings or requirements.

### Conclusion

The `config/` package centralizes and organizes essential configuration settings for your deep learning project. By using this structured approach, you can easily manage and update various aspects of your project, ensuring consistency and flexibility throughout different stages of model development, training, and evaluation.

