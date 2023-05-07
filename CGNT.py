import tensorflow as tf
import numpy as np

# Define the system dynamics and constraints
def state_update(S_curr, I_prev):
    S_max = 100  # maximum value of S, stock quantity
    I_max = 100  # maximum value of I, order quantity
    D = np.random.randint(0, 50)  # random demand
    S = S_curr + I_prev - D  # update state
    if S < 0:
        S = 0
    else:
        S = min(S,S_max)
    return S, D #returns the updated state S

def cost(I, S, D): #cost at a given timestep
    cost = 0
    cost += I*np.random.rand() #random cost per cookie at each timestep
    if D > S+I:
        cost += (D-(S+I))*2 #Additional cost if some customers dont get any cookie
    return cost

# Generate training data
num_timesteps = 10
data = []
state = 100  # initial state
for t in range(num_timesteps):
    demand = np.random.randint(0, 50)
    I = np.random.randint(0, 100)
    cost_t = cost(I, state, demand)
    data.append((state, demand, cost_t, I))
    state, _ = state_update(state, I)
    
# Convert data to numpy array
data = np.array(data)
print(data)

# Define neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

# Define loss function and optimizer
def cost_fn(y_true, y_pred):
    return abs(y_true - y_pred)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# Train the model
x_train = data[:, :3]
print(x_train)
y_train = data[:, 3:]
print(y_train)

model.compile(loss=cost_fn, optimizer=optimizer)
model.fit(x_train, y_train, epochs=100, batch_size=32)

#Geberate some random testing data state, demand, cost
Test_data = np.zeros([100,3])
for i in range(100):
    demand = np.random.randint(0, 50)
    state = np.random.randint(0, 100)
    costt = cost(10, state, demand)
    Test_data[i,:] = [state, demand, costt]

    
print(Test_data)

I = model.predict(Test_data)
I = np.round(I)
print('here comes I bitch',I)
"""
# Convert test data to numpy array
test_data = np.array(test_data)

# Predict optimal values of I
x_test = test_data[:, :2]
y_pred = model.predict(x_test)

# Print predicted values of I
for i in range(num_test_timesteps):
    print("Time step {}: Predicted I = {}".format(i+1, y_pred[i][0]))"""