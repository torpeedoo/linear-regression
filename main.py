import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

def make_data(n_samples, n_features, n_informative, noise, coef, random_state):
    x, y, coef = datasets.make_regression(n_samples=n_samples,#number of samples
                                        n_features=n_features,#number of features
                                        n_informative=n_informative,#number of useful features 
                                        noise=noise,#bias and standard deviation of the guassian noise
                                        coef=coef,#true coefficient used to generated the data
                                        random_state=random_state) #set for same data points for each run

    # Scale feature x (years of experience) to range 0..20
    x = np.interp(x, (x.min(), x.max()), (0, 20))
    # Scale target y (salary) to range 20000..150000 
    y = np.interp(y, (y.min(), y.max()), (1, 1000))
    return (x, y)


def hypothesis_equation(input, params):
    if type(input) == "list":
        outputs = []
        for i in input:
            outputs.append(params[0] + params[1]*input)
        return outputs
    else:
        return params[0] + params[1]*input

def stoch_gradient_descent(learning_rate, x, y, iterations, params):
    new_params = params
    total_its = 0

    #loop through each feature
    for p in range(0, len(params)):
        
        #number of descent iterations
        for i in range(0, iterations):
            total_its = total_its+1
            new_params[p] = new_params[p] - learning_rate*(hypothesis_equation(x[i], new_params) - y[i])*x[i]
            plt.plot(x, y,'.',label='training data')
            plt.plot(x, hypothesis_equation(x, new_params))
            plt.xlabel('x');plt.ylabel('y')
            plt.title('LR')
            plt.savefig('iteration_'+str(total_its))
            plt.clf()

    return new_params


data = make_data(100, 1, 1, 20, True, np.random.randint(1, 100))
x = data[0]
y = data[1]
def_params = [0, 0]

hvals = hypothesis_equation(x, def_params)

new_params = stoch_gradient_descent(0.0002, x, y, 100, def_params)




#model = hypothesis_equation(x, new_params)
#plt.ion() #interactive plot on
#plt.plot(x, y,'.',label='training data')
#plt.plot(x, model)
#plt.xlabel('x');plt.ylabel('y')
#plt.title('LR')
#plt.show()
#plt.pause(60)