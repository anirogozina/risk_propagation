import numpy as np
from scipy.special import expit
from library import GraphElem, Graph, Pipeline, Elem

class CurrentGraph(Graph):
    def get_default_elem(self, time, user_id):
        return Elem(target=None, features=[], time=time, user_id=user_id)
    
class Model:
    def __init__(self, alpha, h, left_bound = 0., right_bound = 1.):
        self.mu = np.random.uniform(-1/100, 1/100, 1)[0]
        self.beta = np.random.uniform(-1/100, 1/100, 1)[0]
        self.w0 = np.random.uniform(-1/100, 1/100, 1)[0]
        self.w1 = np.random.uniform(-1/100, 1/100, 1)[0]
        self.grad = np.array([0., 0., 0., 0.])
        self.features_grad = np.array([0., 0., 0.])
        self.alpha = alpha
        self.features_weights = np.random.uniform(-1/100, 1/100, 3)
        self.h = h
        self.left_bound = left_bound
        self.right_bound = right_bound
        
    def calc_q_t(self, features, prev_p):
        q_t = - self.w0
        for feature_set in features:
            sigma = expit(-self.features_weights[0] + 
                      self.features_weights[1]*feature_set[0] +
                      self.features_weights[2]*feature_set[1])
            q_t += self.w1*sigma
        return expit(q_t)
    
    def predict(self, features, prev_p):
        #print('mu', self.mu)
        #print('beta', self.beta)
        #print('w0', self.w0)
        #print('w1', self.beta)
        return (1 - self.mu)*prev_p + self.beta*(1 - prev_p)*self.calc_q_t(features, prev_p)
    
    def calc_loss(self, elem, prev_p):
        p = self.predict(elem.features, prev_p)
        if elem.target == 1:
            return 5.0*np.log(p)
        if elem.target == 0:
            return np.log(1 - p)
        
    def backprop(self, elem, prev_p):
        if elem.target == 0:
            mult = - 1.0/ (1 - prev_p)
        else:
            mult = 5.0/ prev_p
        mu_grad = -prev_p
        q = self.calc_q_t(elem.features, prev_p)
        beta_grad = (1 - prev_p) * q
        features_grad = np.zeros_like(self.features_weights)
        sigma_sum = 0
        for feature_set in elem.features:
            sigma = expit(-self.features_weights[0] + 
                      self.features_weights[1]*feature_set[0] +
                      self.features_weights[2]*feature_set[1])
            sigma_sum += sigma
            features_grad[0] -= sigma*(sigma - 1)*self.w1
            features_grad[1] += sigma*(sigma - 1)*self.w1*feature_set[0]
            features_grad[2] += sigma*(sigma - 1)*self.w1*feature_set[1]
        w1_grad = self.beta*(1 - prev_p)*q*(1 - q)*sigma_sum
        w0_grad = - self.beta*(1 - prev_p)*q*(1 - q)
        
        
        features_grad *= q*(q - 1)*mult
        new_grad = np.array([mu_grad, beta_grad, w0_grad, w1_grad])*mult
        grad = self.grad * (1 - self.alpha) + new_grad * self.alpha
        self.features_grad = self.features_grad*(1 - self.alpha) + features_grad*self.alpha
        self.mu = min(max(self.left_bound, self.mu + self.h * grad[0]), self.right_bound)
        self.beta = min(max(self.left_bound, self.beta + self.h * grad[1]), self.right_bound)
        self.w0 = self.w0 + self.h * grad[2]
        self.w1 = self.w1 + self.h * grad[3]
        self.w1 = self.w1 + self.h * grad[3]
        #print('features_grad', features_grad)
        self.features_weights += features_grad
        #print('features', self.features_weights)
        