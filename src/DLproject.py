import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import StorePred
from time import perf_counter

seed=1234
tf.random.set_seed(seed)

# Parameters
T = 15 # end time/number of subnetworks
s_0 = 50.
gamma = 1e4 # penalty parameter
smax = 100.
imax = 30.
learning_rate = 0.003
num_epochs = 1000
batch_size = 50
n_it_steps = 20000
N = int(batch_size * n_it_steps / num_epochs) # training set size

# Initial state and labels
inputs = tf.constant(s_0 * np.ones((N,1)), dtype=tf.float32)
outputs = tf.constant(np.zeros_like(inputs))

# Demand
d_array = tf.random.uniform(shape=(N,T), minval=5, maxval=35, dtype=tf.float32)
d_array = tf.floor(d_array)

# Instantiate model
model = StorePred(T, smax, imax, gamma, d_array, batch_size, hidden_units=50)

# Define loss function and optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_function = tf.keras.losses.MeanAbsoluteError()
metric = [tf.keras.metrics.MeanAbsoluteError()]

model.compile(optimizer=optimizer, loss=loss_function, metrics=metric)

# Train the model
start = perf_counter()
history = model.fit(x=inputs, y=outputs, batch_size=batch_size, epochs=num_epochs)
print(f'Computation time = {perf_counter() - start}')

model.example_trajectory(s_0, seed)

# Plot training history
fig, ax = plt.subplots()
ax.semilogy(history.history['mean_absolute_error'])
ax.set_xlabel('Epoch')
ax.set_ylabel('Mean absolute error')
ax.legend()
ax.grid(True)
#fig.savefig('results/loss.png', dpi=300)
plt.show()