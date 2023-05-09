import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def update(s, i, d):
    s_new = s + i - d
    return s_new

def buy_cost(i,t):
    if t <= 5:
        return i
    else:
        return 3*i

def relu(x):
    return tf.nn.relu(x)

def penalty(x, gamma):
    return tf.where(tf.less(x, 0.), gamma * tf.abs(x), 0.)

class StorePred(tf.keras.Model):
    def __init__(self, T, smax, imax, gamma, d_array, batch_size, hidden_units=100):
        super(StorePred, self).__init__()
        self.T = T
        self.smax = smax
        self.imax = imax
        self.gamma = gamma
        self.d_array = d_array
        self.batch_size = batch_size
        self.call_id = 0

        self.network_layers=[]
        for _ in range(T):
            self.network_layers.append(tf.keras.layers.Dense(hidden_units, activation=None, kernel_initializer='random_normal'))
            self.network_layers.append(tf.keras.layers.BatchNormalization())
            self.network_layers.append(tf.keras.layers.Dense(hidden_units, activation=None, kernel_initializer='random_normal'))
            self.network_layers.append(tf.keras.layers.BatchNormalization())
            self.network_layers.append(tf.keras.layers.Dense(1, activation='relu', kernel_initializer='random_normal'))

    def call(self, input, training=False):
        s = input
        cost = tf.zeros(s.shape)
        for t in range(self.T):
            s = tf.reshape(s,(-1,1))
            x = self.network_layers[t*5](s) # input to hidden layer
            x = self.network_layers[t*5+1](x, training=training) # batch norm
            x = relu(x)
            x = self.network_layers[t*5+2](x) # hidden to hidden layer
            x = self.network_layers[t*5+3](x, training=training) # batch norm
            x = relu(x)
            i = self.network_layers[t*5+4](x) # hidden to output layer
            d = tf.reshape(self.d_array[self.call_id*self.batch_size:(self.call_id+1)*self.batch_size,t],s.shape)
            cost += buy_cost(i,t) + penalty(s, self.gamma) + penalty(self.smax - s, self.gamma) \
                    + penalty(i, self.gamma) + penalty(self.imax - i, self.gamma) + penalty(s + i - d, self.gamma)
            s = update(s, i, d)
        cost += penalty(s, self.gamma) + penalty(self.smax - s, self.gamma)
        self.call_id += 1
        return cost
    
    def example_trajectory(self, s_0, seed):
        # tf.random.set_seed(seed)
        # Initial state and labels
        s = tf.constant(s_0 * np.ones((1,1)), dtype=tf.float32)

        # Demand
        d_array = tf.random.uniform(shape=(1,self.T), minval=5, maxval=35, dtype=tf.float32)
        d_array = tf.floor(d_array)

        # Predict
        costs = np.zeros(self.T+1)
        store = np.zeros(self.T+1)
        demand = np.zeros(self.T)
        buy = np.zeros(self.T)

        cost = tf.zeros(s.shape)
        for t in range(self.T):
            s = tf.reshape(s,(-1,1))
            x = self.network_layers[t*5](s) # input to hidden layer
            x = self.network_layers[t*5+1](x, training=False) # batch norm
            x = relu(x)
            x = self.network_layers[t*5+2](x) # hidden to hidden layer
            x = self.network_layers[t*5+3](x, training=False) # batch norm
            x = relu(x)
            i = self.network_layers[t*5+4](x) # hidden to output layer
            # i = tf.floor(i)
            # i = tf.cast(i, tf.float32)
            d = d_array[0,t]
            cost += buy_cost(i,t) + penalty(s, self.gamma) + penalty(self.smax - s, self.gamma) + penalty(i, self.gamma) + penalty(self.imax - i, self.gamma) + penalty(s + i - d, self.gamma)
            costs[t] = cost
            store[t] = s
            demand[t] = d
            buy[t] = i
            s = update(s, i, d)
        store[self.T] = s
        cost += penalty(s, self.gamma) + penalty(self.smax - s, self.gamma)
        costs[self.T] = cost
            

        plt.figure('Cost')
        plt.plot(costs, label='Cost')
        plt.xlabel('Time')
        plt.ylabel('Cost')
        plt.title(f'Cost over time, T={self.T}')
        plt.grid(True)
        plt.savefig(f'results/cost_{seed}_{self.T}.png', dpi=300)

        plt.figure('Store')
        plt.plot(range(self.T+1),store, label='Store')
        plt.plot(range(1,self.T+1),demand, label='Demand')
        plt.plot(buy, label='Buy')
        plt.xlabel('Time')
        plt.ylabel('Store')
        plt.legend()
        plt.title(f'State, action and demand over time, T={self.T}')
        plt.grid(True)
        plt.savefig(f'results/storage_{seed}_{self.T}.png', dpi=300)