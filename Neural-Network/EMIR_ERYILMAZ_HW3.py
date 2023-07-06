import numpy as np
import math
import random
import matplotlib.pyplot as plt

training_set = np.loadtxt("train.csv", delimiter=',', dtype=np.float32, skiprows=1) 
test_set = np.loadtxt("test.csv", delimiter=',', dtype=np.float32, skiprows=1)

def single_layer(training_set, test_set):
    weights = []
    weights.append(random.random()*0.02-0.01)
    epochs = 10
    learning_rate = 0.0021

    for input in training_set[0][:-1]:
        weights.append(random.random()*0.02-0.01)
    #Training
    epoch_error = []
    for epoch in range(epochs):
        for row in range(len(training_set)):
            dw= []
            y = weights[1]*training_set[row][0] + weights[0] #We take the row +1st index for the weight because w0 is the bias
            error = 0.5*(training_set[row][0]-y)**2
            #print(error)
            delta = learning_rate*(training_set[row][0]-y)
            dw.append(delta)
            delta = learning_rate*(training_set[row][0]-y)*training_set[row][0]
            dw.append(delta)
            for index,weight in enumerate(weights):
                weights[index] += dw[index]  
        
        #Calculate the error in the current epoch
        error_current = 0
        for row in range(len(training_set)):
            y_value = weights[1]*training_set[row][0] + weights[0]
            x_value = training_set[row][0]
            error_current += (0.5*((y_value-training_set[row][1])**2))
        epoch_error.append(error_current)
    epoch_list = []
    for epoch in range(epochs):
        epoch_list.append(epoch)
    
    """plt.scatter(epoch_list,epoch_error)
    plt.show()
    """

    #Testing
    error = 0
    outputs = []
    for row in test_set:
        error += 0.5*((row[0]*weights[1]+weights[0]-row[1])**2)
        outputs.append(weights[1]*row[0]+weights[0])
    plt.scatter(test_set[:,0],test_set[:,1])
    plt.plot(test_set[:,0],outputs)

    plt.show()
    #print(error)
    return error


def multi_layer(training_set, test_set, n_hidden_units):
    weights_1 = []
    weights_2 = []

    bias_1 = []

    epochs = 1000
    learning_rate = 0.021

    for input in range(n_hidden_units):
        weights_1.append(random.random()*0.02-0.01)
    for input in range(n_hidden_units):
        bias_1.append(random.random()*0.02-0.01)
    for input in range(n_hidden_units):
        weights_2.append(random.random()*0.02-0.01)

    bias2 = random.random()*0.02-0.01
    
    #Training

    epoch_error = []
    for epoch in range(epochs):     
        for row in range(len(training_set)):
            z = []
            dv = []
            dw = []
            dw_bias = []
            y = bias2
            for h_unit in range(n_hidden_units):
                z_current = weights_1[h_unit]*training_set[row][0]+bias_1[h_unit] #w*x+w0
                z.append(1/(1+math.exp(-z_current)))    
            for i in range(len(z)):
                y+=z[i]*weights_2[i]
                error = 0.5*((training_set[row][1]-y)**2)
                delta = learning_rate*(training_set[row][1]-y)*z[i]
                dv.append(delta)

            delta_bias = learning_rate*(training_set[row][1]-y)
            
             
            for h_unit in range(n_hidden_units):
                delta = learning_rate*(training_set[row][1]-y)*weights_2[h_unit]*z[h_unit]*(1-z[h_unit])*training_set[row][0]
                dw.append(delta)
            for h_unit in range(n_hidden_units):
                delta_bias = learning_rate*(training_set[row][1]-y)*weights_2[h_unit]*z[h_unit]*(1-z[h_unit])
                dw_bias.append(delta_bias)

            for i in range(len(weights_2)):
                weights_2[i] += dv[i] 

            for i in range(len(weights_1)):
                weights_1[i] += dw[i]
                bias_1[i] += dw_bias[i]
            bias2 += delta_bias

        #Calculate error in current epoch
        error_current = 0
        for row in range(len(training_set)):
            y = bias2
            h_units = []
            for h_unit in range(n_hidden_units):
                sigmoid_input = (training_set[row][0]*weights_1[h_unit]+bias_1[h_unit])
                h_units.append(1/(1+math.exp(-sigmoid_input))) 
            for h_unit in range(n_hidden_units):
                y+=(weights_2[h_unit]*h_units[h_unit])
            error_current += 0.5*((training_set[row][1]-y)**2)
        epoch_error.append(error_current)
    
    epoch_list = []
    for epoch in range(epochs):
        epoch_list.append(epoch)
    #plt.scatter(epoch_list,epoch_error)
    #plt.show()

    error_test = 0
    outputs = []
    for row in range(len(test_set)):
        y = bias2
        h_units = []
        for h_unit in range(n_hidden_units):
            sigmoid_input = (test_set[row][0]*weights_1[h_unit]+bias_1[h_unit])
            h_units.append(1/(1+math.exp(-sigmoid_input))) 
        for h_unit in range(n_hidden_units):
            y+=(weights_2[h_unit]*h_units[h_unit])
        error_test += 0.5*((test_set[row][1]-y)**2)
        outputs.append(y)
    curve = np.polyfit(test_set[:,0],outputs,6)
    poly = np.poly1d(curve)
    y_values = []
    for i in range(len(test_set)):
        y_values.append(poly(test_set[i][0]))

    xs, ys = zip(*sorted(zip(test_set[:,0], y_values)))
    
    plt.scatter(test_set[:,0], test_set[:,1])
    #print(outputs)
    plt.plot(xs,ys)
    plt.show()
    return error_test/len(test_set)

x_vals = [0,10,20,50]
y_vals = []

y_vals.append(single_layer(training_set=training_set,test_set=training_set))
y_vals.append(multi_layer(training_set=training_set,test_set=training_set,n_hidden_units=10))
y_vals.append(multi_layer(training_set=training_set,test_set=training_set,n_hidden_units=20))
y_vals.append(multi_layer(training_set=training_set,test_set=training_set,n_hidden_units=50))

plt.scatter(x_vals,y_vals)
plt.plot(x_vals,y_vals)
plt.show()


plt.scatter(x_vals[1:],y_vals[1:])
plt.plot(x_vals[1:],y_vals[1:])
plt.show()


x_vals = [0,10,20,50]
y_vals = []

y_vals.append(single_layer(training_set=training_set,test_set=test_set))
y_vals.append(multi_layer(training_set=training_set,test_set=test_set,n_hidden_units=10))
y_vals.append(multi_layer(training_set=training_set,test_set=test_set,n_hidden_units=20))
y_vals.append(multi_layer(training_set=training_set,test_set=test_set,n_hidden_units=50))

plt.scatter(x_vals, y_vals)
plt.plot(x_vals,y_vals)
plt.show()


plt.scatter(x_vals[1:],y_vals[1:])
plt.plot(x_vals[1:],y_vals[1:])
plt.show()