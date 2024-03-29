# Linear Regression

Linear regression is the process of mapping a linear funciton to a dataset; by doing this we can predict unknown values with a good ammount of accuracy. When fist designing our learning algorithm we need to figure out how to represent the hypothesis **h**, which is the prediction function. In Linear Regression we can represent the hypothesis with the following equation:

$$h(x) = \theta_0 + \theta_1x$$

The goal of our learning algorithm is to minimize our cost function. In this case it can be represented with this equation 

$$J(\theta) = \frac{1}{2} \sum _{i=1}^m (h(x^{(i)}) - y^{(i)})^2$$

This equation represents the sum of all the predicted values $h(x)$ minus the actual values $y$ in the entire dataset. We can then change theta to reduce the output of the function. The half constant is only to make the math easier down the line. 

## How can we optimize the cost function?

To optimize the loss function we want to change our theta values so that the loss function can be minimized. This can be done through the use of gradient descent, below we can see a graph of the loss function based on theta one and two. Generally Gradient Descent takes a step in the steepest direction downward each iteration, hopefully ending in the lowest point.


![](/rsr/GD.PNG)
 
We can formally define our gradient descent function as follows:

$$(\theta_j := \theta_j - \alpha \sum_{i=1}^m (h(x^{(i)})-y^{(i)})*x_j^{(i)})_{(j=1...n)}$$

where $(h(x)-y)*x_j$ is the partial derivative of $J(\theta)$ with respect to $\theta_j$. we repeat this update of $\theta_j$ until convergence.

**If our learning rate $\alpha$ is too large me may overshoot the minimum. On the other hand if it is too small it will be very slow.**

This is known as **batch** gradient descent which is great in the case of smaller data sets but falls short when used with large data sets.

Another option when using large datasets would be to use **stochastic** gradient descent which instead of iterating over the loss function for all the elements in the set we instead only use one example and iterate over that example each step in the process. which would give us the following equation:

$$(\theta_j := \theta_j - \alpha(h(x^{(i)})-y^{(i)})*x_j^{(i)})_{(j=1...n)}$$

The flaw of this method is that it will never truly converge, leaving us with a less than optimal hypothesis equation. But this is the more common method just because of how slow batch Gradient Descent is in practice. Using stochastic and also decreasing the learning rate with each iteration also works well because it will bounce around a smaller area in the end, making for a more accurate result.

## Implementing these concepts into python

The first step I took when implementing these principles into python was to define a function holding our hypothesis equation.
```python
def hypothesis_equation(input, params):
    if type(input) == "list":
        outputs = []
        for i in input:
            outputs.append(params[0] + params[1]*input)
        return outputs
    else:
        return params[0] + params[1]*input
```

My next step was to define my gradient descent function. This includes code to plot the hypothesis function with the new parameters so we can make an animation later.
```python
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
```
Finally, all we have to do is to run these function to train our algorithm.
```python
data = make_data(100, 1, 1, 20, True, np.random.randint(1, 100))
x = data[0]
y = data[1]
def_params = [0, 0]

hvals = hypothesis_equation(x, def_params)

new_params = stoch_gradient_descent(0.0002, x, y, 100, def_params)
```
When we run this we get a set of images we can turn into an animation.

![](/rsr/LR_anim.gif)

## Resources
[Stanford CS229](https://youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU&si=jh4EJ8Pf7Z7miBP8)

