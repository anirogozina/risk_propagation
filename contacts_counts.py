import numpy as np
from scipy.special import expit
from library import GraphElem, Graph, Pipeline, Elem

class CurrentGraph(Graph):
    def get_default_elem(self, time, user_id):
        return Elem(target=None, features=[0], time=time, user_id=user_id)
    
    
class Model:
    def __init__(self, alpha, h, left_bound = 0., right_bound = 1.):
        self.mu = np.random.uniform(-1/100, 1/100, 1)[0]
        self.beta = np.random.uniform(-1/100, 1/100, 1)[0]
        self.w0 = np.random.uniform(-1/100, 1/100, 1)[0]
        self.w1 = np.random.uniform(-1/100, 1/100, 1)[0]
        self.grad = np.array([0., 0., 0., 0.])
        self.alpha = alpha
        self.h = h
        self.left_bound = left_bound
        self.right_bound = right_bound
    
    def predict(self, features, prev_p):
        return (1 - self.mu)*prev_p + self.beta*(1 - prev_p)*expit(self.w1 * features[0] - self.w0)
    
    def calc_loss(self, elem, prev_p):
        p = self.predict(elem.features, prev_p)
        if elem.target == 1:
            return 5*np.log(p)
        if elem.target == 0:
            return np.log(1 - p)
        
    def backprop(self, elem, prev_p):
        if elem.target == 0:
            mult = - 1.0 / (1 - prev_p)
        else:
            mult = 5.0 / prev_p
        mu_grad = -prev_p
        q = expit(self.w1 * elem.features[0] - self.w0)
        beta_grad = (1 - prev_p) * q
        w1_grad = self.beta*(1 - prev_p)*q*(1 - q)*elem.features[0]
        w0_grad = - self.beta*(1 - prev_p)*q*(1 - q)
        
        new_grad = np.array([mu_grad, beta_grad, w0_grad, w1_grad])*mult
        grad = self.grad * (1 - self.alpha) + new_grad * self.alpha
        self.mu = min(max(self.left_bound, self.mu + self.h * grad[0]), self.right_bound)
        self.beta = min(max(self.left_bound, self.beta + self.h * grad[1]), self.right_bound)
        self.w0 = self.w0 + self.h * grad[2]
        self.w1 = self.w1 + self.h * grad[3]
        
        