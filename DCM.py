
import numpy as np
import pandas as pd
from scipy import optimize


def uCRRA(x,coef):
        return x**(1-coef)/(1-coef)

class choiceModel:

    '''
        Estimating a logit model using data collected by the staircase method

        1. utility function takes the CRRA format: u(x) = x^(1- riskCoef) / (1 - riskCoef)
        Note riskCoef ≠ 1. I suggest using 2 as the initial value.
        
        2. choice involves two options
            option A: get reward x with prob. delta
            option B: get reward c for sure
        The amount c is constant, but x is a variable

        For intertemporal choice, A is LL and B is SS. delta can be the discount factor.

        3. choice probability for option A: 
            P(A) = sigmoid( u_diff / temp )
        where u_diff = delta * u(x) - u(c), temp is the temperature parameter
    '''

    def __init__(self,data,x,choice,fixed_args):
        self.data = data
        self.x = x
        self.choice = choice
        self.fixed_args = fixed_args

    '''
    x: str or list
    Specify the names of columns which will be used as x

    choice: str or list
    Specify the names of columns which will be used as observed choices
    Each choice must be binary (0 or 1). 1 = choosing A; 0 = choosing B.

    fixed_args: dict
    Fixed arguments, must include c. For example, {'c':30, 'delta': 0.5} 
    '''

    def set_init_param(self, param_init=None, param_keys=None, latent_keys=None):

        param_setup = {
                "param_init": param_init,
                "param_keys": param_keys,
                "latent_keys": latent_keys
            }
        
        for key, value in param_setup.items():
            if value is not None:
                setattr(self, key, np.array(value))
        
        if param_keys is not None and param_init is not None:
            print('Initial parameters:', dict(zip(param_keys, param_init)))
            if len(param_keys) == len(param_init):
                return "The key list and value list of parameters have mismatched lengths"
         
    '''
    param_keys: list 
    Set up the keys for the parameters.

    param_init: list 
    Set up the intial values for the parameters. Must be corresponding to param_keys.

    latent_keys: list
    Set up the keys for latent variables
    '''

    @staticmethod
    def logitProb(x_1,x_2,riskCoef,temp,delta):

        # choice probability for a specific choice question

        u_diff = delta*uCRRA(x_1,riskCoef) - uCRRA(x_2,riskCoef)

        prob = 1/(1+np.exp(-u_diff/temp))

        return prob
    

    def logLike(self, coefs):

        all_args = self.fixed_args | dict(zip(self.param_keys, coefs))
        
        x = self.data[self.x]
        c = all_args['c']
        delta = all_args['delta']
        riskCoef = all_args['riskCoef']
        temp = all_args['temp']

        p = self.logitProb(x,c,riskCoef,temp,delta).values
        y = self.data[self.choice].values
        
        logLogit = y*np.log(p) + (1-y)*np.log(1-p)

        if len(logLogit.shape) > 1:
            logLogit = logLogit.sum(axis = 1)

        return logLogit.sum()
        

    def fit_param(self):

        obj = lambda coefs: - self.logLike(coefs)

        x0 = list(self.param_init)
        bounds = [(None,None)] * len(self.param_init)

        self.result = optimize.minimize(obj, x0, method='L-BFGS-B', bounds=bounds)

        return self.result
    
    

class mixedModel(choiceModel):

    def likeIndiv(self,latent_class: dict):
        
        choices = self.data[self.choice].values
        x = self.data[self.x].values
        c = self.fixed_args['c']
        delta = self.fixed_args['delta']
        riskCoef = np.array(latent_class['riskCoef'])[:,np.newaxis]
        temp = np.array(latent_class['temp'])[:,np.newaxis]

        prob = np.ones((len(temp),len(x)))

        for i in range(x.shape[1]):
            p = self.logitProb(x[:,i],c,riskCoef,temp,delta)
            y = choices[:,i]
            logLogit = y*p + (1-y)*(1-p)
            prob = prob*logLogit

        return prob

 
    def postProblatent(self, latent_class: dict, latent_share: list):

        # m × n matrix
        # m: number of latent classes
        # n: number of individuals
        choiceProb = self.likeIndiv(latent_class)

        latent_share = np.array(latent_share)
        
        jointProb = choiceProb * latent_share[:, np.newaxis]

        # Conditional on person i choosing j, the probability that the person is in latent class b
        post_prob = jointProb / jointProb.sum(axis=0, keepdims=True)

        return post_prob
    

    def updateShare(self, latent_class: dict, latent_share: list):

        post_prob = self.postProblatent(latent_class, latent_share)
        new_share = post_prob.sum(axis=1) / post_prob.sum()

        return new_share
    
    


        














    

    def logLikeMSL(self,params):

        param_keys = self.init_params.keys()
        params = dict(zip(param_keys, params))

        mean = [params['mean_risk'], params['mean_temp']]
        covar = [
                    [params['std_risk']**2, 
                    params['rho']*params['std_risk']*params['std_temp']], 
                    [params['rho']*params['std_risk']*params['std_temp'], 
                    params['std_temp']**2]
                ]
        
        random_vars = []

        n_obs = len(self.data) 

        while len(random_vars) < n_obs*n_draws:
            sample = np.random.multivariate_normal(mean, covar, size=1)
            
            # Check if both values are positive
            if sample[0][0] > 0 and sample[0][1] > 0:
                random_vars.append(sample[0])

        random_vars = np.array(random_vars)

        riskCoef = random_vars[:,0]
        temp = random_vars[:,1]
        c = self.fixed_args['c']
        delta = self.fixed_args['delta']
        
        x = np.tile(self.data[self.x].values, (n_draws, 1))
        y = np.tile(self.data[self.choice].values, (n_draws, 1))

        p = self.logitProbMSL(x,c,riskCoef,temp,delta)

        logLogit = y*np.log(p) + (1-y)*np.log(1-p)

        return - logLogit.sum() / n_draws
    
    