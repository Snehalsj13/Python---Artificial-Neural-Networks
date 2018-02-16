
# coding: utf-8

# In[18]:


import numpy as np
from matplotlib import pyplot as plt


# In[19]:


def add_noise(y):
    """
    Adds random noise to a set of points
    """
    noise_func = np.vectorize(lambda x: x + np.random.uniform(-np.max(y)/4, np.max(y)/4, 1))
    return noise_func(y)


# In[20]:


def generate_points(start, stop, number, a=3.65, b=6.7, c=12.8):
    """
    Generates points along a line with the equation
    y = mx + b (parent function)
    """
    x = np.linspace(start, stop, num=number)
    quadratic_function = np.vectorize(lambda x: a*x**2 + b*x + c)
    y = quadratic_function(x)
    return (x, y)


# In[21]:


def get_y_values(X, a=None, b=None, c=None):
   
    return a*X**2 + b*X + c


# In[22]:


## Generate 20 points on a line with x values between 1 and 30
## with m = 0.6 and b = 30
X, _Y = generate_points(1, 30, 100, a=0.6, b=10, c=30)
## Add random noise to the y values
Y = add_noise(_Y)


# In[23]:


plt.scatter(X, Y)


# In[30]:


def get_linear_gradients_mse(X, Y, Y_hat):
    """
    returns gradients for the two variables m and b assuming mean square error
    for a linear equation y = mx + b
    """
    # dL/dm
    a_gradient = np.sum(2*X**2*(Y_hat - Y), dtype=np.int32)/len(Y) ### Try removing the normalization factor len(Y)
    # dL/db
    b_gradient = np.sum(2*X*(Y_hat - Y), dtype=np.int32)/len(Y)
   
    c_gradient = np.sum(2*(Y_hat - Y))/len(Y)
    return a_gradient, b_gradient, c_gradient


# In[41]:


# Initial Values
a = 3.65
b = 6.7
c = 12.8
# Learning rate
alpha = 0.0000001
# Number of iterations
N = 100
for i in range(N):
    Y_hat = get_y_values(X, a, b, c)
    a_grad, b_grad, c_grad = get_linear_gradients_mse(X, Y, Y_hat)
    # update m and b for the next iteration
    a = a - alpha * a_grad
    b = b - alpha * b_grad
    c = c - alpha * c_grad
# The final value of 
plt.plot(X, get_y_values(X, a, b, c), 'r')
plt.scatter(X, Y)
print "Estimates after {} iterations: a = {}, b = {}, c = {}".format(N, a, b, c)

