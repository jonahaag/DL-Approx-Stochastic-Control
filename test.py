import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def update(s, i, d):
    s_new = s + i - d
    return s_new

def buy_cost(i,t):
    return t*i

def relu(x):
    return tf.nn.relu(x)

def penalty(x):
    return tf.where(tf.less(x, 0.), gamma * tf.square(x), 0.)

class Subnetwork(tf.keras.Model):
    def __init__(self, batch_size):
        super(Subnetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(100, activation=None, kernel_initializer='random_normal') # , input_shape=(batch_size,)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(100, activation=None, kernel_initializer='random_normal')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.dense_out = tf.keras.layers.Dense(1, activation=None, kernel_initializer='random_normal')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.bn1(x, training=training)
        x = relu(x)
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        x = relu(x)
        outputs = self.dense_out(x)
        return outputs

class MyModel(tf.keras.Model):
    def __init__(self, T, K, mu, sigma, batch_size):
        super(MyModel, self).__init__()
        self.subnetworks = [Subnetwork(batch_size) for _ in range(T)]
        self.T = T
        self.K = K
        self.mu = mu
        self.sigma = sigma

    def call(self, input, training=False):
        s = input[:,0]
        d_batch = input[:,1:]
        s = tf.reshape(s,(-1,1))
        i_prev = self.subnetworks[0](s, training=training)
        cost = tf.zeros(s.shape)

        for t in range(1, self.T):
            d = d_batch[:,t]
            # d = tf.random.normal(s.shape, mean=self.mu, stddev=self.sigma)
            s = update(s, i_prev, d)
            i = self.subnetworks[t](s, training=training)
            cost += buy_cost(i,t)
            cost += penalty(s)
            cost += penalty(smax - s) 
            cost += penalty(i)
            cost += penalty(imax - i) 
            cost += penalty(s + i - d)
            print(cost)
            i_prev = i

        return cost
    
    def simulate_one_example(self, input):
        s = input[:,0]
        d_batch = input[:,1:]
        s = tf.reshape(s,(-1,1))
        i_prev = self.subnetworks[0](s, training=False)
        
        s_history = np.zeros((s.shape[0],T))
        i_history = np.zeros((s.shape[0],T))
        cost_history = np.zeros((s.shape[0],T))

        s_history[:,0] = s.numpy().reshape(-1,)
        i_history[:,0] = i_prev.numpy().reshape(-1,)
        cost_history[:,0] = 0.

        cost = tf.zeros(s.shape)

        for t in range(1, self.T):
            d = d_batch[:,t]
            # d = tf.random.normal(s.shape, mean=self.mu, stddev=self.sigma)
            s = update(s, i_prev, d)
            print("s: ", s.shape)
            i = self.subnetworks[t](s, training=False)
            cost += buy_cost(i,t)
            cost += penalty(s)
            cost += penalty(smax - s) 
            cost += penalty(i)
            cost += penalty(imax - i) 
            cost += penalty(s + i - d)
            i_prev = i
            s_history[:,t] = s.numpy().reshape(-1,)
            i_history[:,t] = i_prev.numpy().reshape(-1,)
            cost_history[:,t] = cost.numpy().reshape(-1,)
        return s_history, i_history, cost_history

    # @tf.function
    # def train_step(self, s_0):
    #     with tf.GradientTape() as tape:
    #         cost = self.call(s_0, training=True)
    #     gradients = tape.gradient(cost, model.trainable_variables)
    #     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    #     return cost

tf.random.set_seed(42)

T = 10 # number of subnetworks
mu = 5.0 # mean of d_t
sigma = 1.0 # standard deviation of d_t
s_0 = 50. # initial value of s_0
K = 1.0 # constant in the cost function
gamma = 1000. # penalty parameter
smax = 100. # maximum value of s_t
imax = 10. # maximum value of i_t
learning_rate = 0.001 # learning rate for SGD
num_epochs = 1000
batch_size = 10
n_it_steps = 15000
N = int(batch_size * n_it_steps / num_epochs) # training set size

inputs = tf.constant(s_0 * np.ones((N,1)), dtype=tf.dtypes.float32)
outputs = tf.constant(np.zeros_like(inputs))
d_array = tf.random.normal([N,T], mean=mu, stddev=sigma)
inputs = tf.concat([inputs,d_array],1)

model = MyModel(T, K, mu, sigma, batch_size)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_function = tf.keras.losses.MeanSquaredError()
metric = [tf.keras.metrics.MeanSquaredError()]

model.compile(optimizer=optimizer, loss=loss_function, metrics=metric)

history = model.fit(x=inputs,
                y=outputs,
                batch_size=batch_size, 
                epochs=num_epochs)

model.save('saved_model/my_model')

# s_history, i_history, cost_history = model.simulate_one_example(inputs[:10,:])
# plt.plot(range(T),s_history[0,:])
# plt.plot(range(T),i_history[0,:])
# plt.plot(range(T),inputs[0,1:])
# plt.plot(range(T),cost_history[0,:])
# plt.show()

# cost_history = []
# for epoch in range(num_epochs):
#     cost = model.train_step(s_0, mu, sigma)
#     cost_history.append(cost)
#     print(f'Epoch {epoch + 1} cost: {cost:.4f}')


