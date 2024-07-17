from tensorflow.keras.optimizers import Adam
from Config.hyperparameters import learning_rate

optimizer = Adam(learning_rate=learning_rate)
loss_function = 'sparse_categorical_crossentropy'
metrics = ['accuracy']