import math
import random

random.seed(42) #setting the seed for reproducibility
w = [ [random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)], 
    [random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)] ] #input weights
v = [random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)] #hidden layer weights
w0 = [0,0] #input layer bias
v0 = 0 #hidden layer bias

x = [ [0,0],[0,1],[1,0],[1,1] ]
target  = [1,0,0,1] #XNOR

print('Training.......')
alpha = 1

z = [0,0]  #hidden layer outputs
gradient = [0,0] #hidden layer gradients

count = 0
while count < 50000:
    output = []
    for i in range(len(x)): #this represents trianing pair
        a1, a2 = x[i]
        t = target[i]

        #Forward Propagation
        for j in range(len(z)):   #calculating hidden layer outputs
            Zin = w0[j] + a1*w[0][j] + a2*w[1][j]
            z[j] = 1/(1+math.exp(-Zin))

        #output layer computations
        Yin = v0 + z[0] * v[0] + z[1] * v[1]
        y = 1/(1+math.exp(-Yin)) #calculating final output
        output.append(y)

        #Backpropagation
        error = (t - y)*y*(1-y)  #calculating error gradient for output neuron
        v0 = v0 + alpha * error

        for j in range(len(z)):
            gradient[j] = z[j]*(1-z[j])*error*v[j]  #calculating gradient for hidden layer
            v[j] = v[j] + alpha*error*z[j]          #updating hidden layer weights
            w[0][j] = w[0][j] + alpha*gradient[j]*a1    #updating weights of x1
            w[1][j] = w[1][j] + alpha*gradient[j]*a2    #updating weights of x2
            w0[j] = w0[j] + alpha * gradient[j]

    count += 1


print('Trained')
print('Testing')
ch = 'y'

while ch.lower() == 'y':
    x1 = int(input('Enter the first input: '))
    x2 = int(input('Enter the second input: '))

    for j in range(len(z)):   #calculating hidden layer outputs
        Zin = w0[j] + x1*w[0][j] + x2*w[1][j]
        z[j] = 1/(1+math.exp(-Zin))

    #output layer computations
    Yin = v0 + z[0] * v[0] + z[1] * v[1]
    y = 1/(1+math.exp(-Yin)) #calculating final output
    print('Output is: ',round(y)) #rouding as it is easy to interpret

    ch = input('Want to do more? (y/n) ')


    

