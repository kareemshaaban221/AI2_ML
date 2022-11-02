# for statistics calculations
import math

# summition of x*p(x) 
## we will assume that expectation = mean
def expectation(data):
    return sum(data) / len(data)

# (standard deviation)^2 = V(X) = E(X^2) - E(X)^2
# Sum Of (X-mue)^2 / len(X)
def variance(data, ddof=1): # column
    #! E(X^2) - E(X)^2
    return  sum([ (x - expectation(data)) ** 2 for x in data]) / (len(data) - ddof)

# sqrt(V(X))
def standard_deviation(data):
    return math.sqrt(variance(data))

# e^-0.5((x-exp)/std) / (std*sqrt(2*pi))
def standard_normal_distribution(data):
    std = standard_deviation(data)
    exp = expectation(data)
    var = variance(data)
    constant = ( std * math.sqrt(2 * math.pi) ) ** -1
    
    return [constant * math.e ** -((x-exp) ** 2 / (2*var)) for x in data]