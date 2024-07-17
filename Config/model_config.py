from tensorflow.keras.layers import Dense, Dropout

# Example model architecture configuration
model_layers = [
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
]