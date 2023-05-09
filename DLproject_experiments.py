import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import StorePred
from time import perf_counter

histories = np.zeros((5,1000,3))

for i, T in enumerate([10, 15, 20]):
    for seed in range(5):
        tf.random.set_seed(seed)

        # T = 10 # number of subnetworks

        s_0 = 50. # initial value of s_0
        gamma = 1e4 # penalty parameter
        smax = 100. # maximum value of s_t
        imax = 30. # maximum value of i_t
        learning_rate = 0.003 # learning rate for SGD
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
        print(f'Seed = {seed}, T = {T}, time = {perf_counter() - start}')
        histories[seed, :, i] = np.reshape(history.history['mean_absolute_error'],(1,1000))

        model.example_trajectory(s_0, seed)

# Plot training history
fig, ax = plt.subplots()
plt.figure('Loss', figsize=(15, 10))
plt.semilogy(histories[:,:,0].mean(axis=0), label='T=10')
plt.fill_between(np.arange(num_epochs), histories[:,:,0].mean(axis=0)-histories[:,:,0].std(axis=0), histories[:,:,0].mean(axis=0)+histories[:,:,0].std(axis=0), alpha=0.3)
plt.semilogy(histories[:,:,1].mean(axis=0), label='T=15')
plt.fill_between(np.arange(num_epochs), histories[:,:,1].mean(axis=0)-histories[:,:,1].std(axis=0), histories[:,:,1].mean(axis=0)+histories[:,:,1].std(axis=0), alpha=0.3)
plt.semilogy(histories[:,:,2].mean(axis=0), label='T=20')
plt.fill_between(np.arange(num_epochs), histories[:,:,2].mean(axis=0)-histories[:,:,2].std(axis=0), histories[:,:,2].mean(axis=0)+histories[:,:,2].std(axis=0), alpha=0.3)
plt.xlabel('Epoch')
plt.ylabel('Mean absolute error')
plt.legend()
plt.grid(True)
plt.savefig('results/loss_five_seeds_three_times.png', dpi=300)