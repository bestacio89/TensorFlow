# Import necessary modules and configurations
from keras.api.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from Config.hyperparameters import batch_size, epochs
from Config.paths import data_dir, saved_models_dir, logs_dir
from Config.model_config import model_layers
from Config.training_config import optimizer, loss_function, metrics
from Data.datasets import load_data
from Utils.helpers import plot_history, save_model


def main():
    # Load and preprocess data
    train_generator, validation_generator = load_data(data_dir)

    # Build model using configurations from model_config.py
    model = Sequential(model_layers)
    model.compile(optimizer=optimizer,
                  loss=loss_function,
                  metrics=metrics)

    # Train model
    history = model.fit(
        train_generator,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_generator
    )

    # Save trained model
    model.save(saved_models_dir + 'my_model')

    # Logging
    with open(logs_dir + 'training.log', 'a') as f:
        f.write('Training completed.\n')

    print("Training completed successfully.")

    # Optional: Plot training history
    plot_history(history)


if __name__ == "__main__":
    main()
