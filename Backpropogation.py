
# coding: utf-8

# In[49]:


def forward(x, y, z, w1, w2, w3):
    x = x * w1
    y = y * w2
    z = z * w3
    return z*(x + y)


# In[50]:


def backward(x, y, z, w1, w2, w3):
    dfdw1 = x*z*w3 
    dfdw2 = y*z*w3
    dfdw3 = x*z*w1 + y*z*w3
    return dfdw1, dfdw2, dfdw3


# In[53]:


x = 2
y = 5
z = 9
w1 = 0.5
w2 = 0.45
w3 = 0.8
dw1, dw2, dw3 = backward(x, y, z, w1, w2, w3)
print ("Gradients on X: {}, Y: {} and Z: {}".format(dw1, dw2, dw3))


# In[52]:


def backward_w_loss(V, x, y, z):
    dldw1 = -2 * x * z * w3 * (V - (x * w1 + y * w2) * z * w3)
    dldw2 = -2 * z * y * w3 * (V - (x * w1 + y * w2) * z * w3)
    dldw3 = -2 * (x * z * w1 + y * z * w3) * (V - (x * w1 + y * w2) * z * w3)
    return dldw1, dldw2, dldw3


# In[60]:


x = 12.3791
y = 1.4782
z = 8.192
w1 = 0.2365
w2 = 2
w3 = 3
alpha = 0.00000000001
final_value = 144
n = 1000
for i in range(n):
    dw1, dw2, dw3 = backward_w_loss(final_value, x*w1, y*w2, z*w3)
    w1 = w1 - alpha * dw1
    w2 = w2 - alpha * dw2
    w3 = w3 - alpha * dw3
    print(w1, w2, w3)
print ("Final Values of w1: {}, w2: {} and w3: {}".format(w1, w2, w3))
# forward is only used here below
print ("Evaluation of (x*w1 + y*w2) * z*w3 = {}".format(forward(x, y, z, w1, w2, w3)))

