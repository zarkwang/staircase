
import numpy as np
import pandas as pd
import itertools
import multiprocessing as mp  
from scipy import stats
from scipy import optimize
from tqdm import tqdm
from functools import partial


class staircase:

    def __init__(self,func_params,x_params,n_subjects=100,
                 *args, **kwargs):         
        
        self.func_params = func_params
        self.x_init = x_params['x_init']
        self.x_step_up = x_params['x_step_up']
        self.x_step_down = x_params['x_step_down']
        self.n_subjects = n_subjects


    @staticmethod
    def choice_gen_func(x,theta,temp):

        diff = temp*(0.5*x**theta - 25**theta) 

        prob = 1/(1+np.exp(-diff))

        choice = np.random.choice([1,0],size=1,p=[prob,1-prob])

        return choice[0]
    

    def run(self,n_iter=1000):

        result_cols = ['iter','prop_up','prop_init','prop_down','p_fisher']

        result = {col:[] for col in result_cols}

        for i in range(n_iter):

            _theta = np.random.normal(loc=self.func_params['mean_theta'],
                                          scale=self.func_params['sd_theta'],
                                          size=self.n_subjects)

            choice_list = [self.choice_gen_func(self.x_init,theta,self.func_params['temp']) for theta in theta_list]

            idx_choice_1 = np.where(np.array(choice_list) == 1)[0]
            idx_choice_0 = np.where(np.array(choice_list) == 0)[0]

            choice_down_list = [self.choice_gen_func(self.x_init - self.x_step_down, theta, self.func_params['temp']) 
                                    for theta in theta_list[idx_choice_1]]
            
            choice_up_list = [self.choice_gen_func(self.x_init + self.x_step_up, theta, self.func_params['temp']) 
                                    for theta in theta_list[idx_choice_0]]

            n_choice_1 = len(idx_choice_1)
            n_choice_0 = len(idx_choice_0)
            n_down_choice_1 = sum(choice_down_list)
            n_down_choice_0 = n_choice_1 - n_down_choice_1
            n_up_choice_1 = sum(choice_up_list)
            n_up_choice_0 = n_choice_0 - n_up_choice_1

            # 2x2 contingency table [ [a, b], [c, d] ]
            # where 'a' and 'b' are counts for one group, and 'c' and 'd' for another.
            # tab_1 = np.array([[n_up_choice_1,n_up_choice_0],
            #                 [n_choice_1,n_choice_0]])

            # tab_2 = np.array([[n_choice_1,n_choice_0],
            #                 [n_down_choice_1,n_down_choice_0]])
            
            tab_3 = np.array([[n_up_choice_1,n_up_choice_0],
                            [n_down_choice_1,n_down_choice_0]])
            
            odds_1, p_1 = stats.fisher_exact(tab_3, alternative='less')

            # odds_1, p_1 = stats.fisher_exact(tab_1, alternative='less')
            # odds_2, p_2 = stats.fisher_exact(tab_2, alternative='less')

            result['iter'] += [i]
            result['prop_up'] += [n_up_choice_1/n_choice_0]
            result['prop_init'] += [n_choice_1/self.n_subjects]
            result['prop_down'] += [n_down_choice_1/n_choice_1]
            result['p_fisher'] += [p_1]
            # result['p_fisher_down'] += [p_2]
            # result['p_fisher_total'] += [min(min(sorted([p_1,p_2])*np.array([2,1])),1)]


        self.result = pd.DataFrame(result)
        self.prop_reject_null = sum(self.result['p_fisher'] < 0.05) / n_iter




# simulation function
def runSimu(i,df_params):

    func_params = df_params[['mean_theta','sd_theta','temp']].loc[i]
    x_params = df_params[['x_init','x_step_up','x_step_down']].loc[i]
    n_ = df_params['n_subject'].iloc[i]

    simu = staircase(func_params,x_params,n_subjects=n_)
    simu.run()

    return (i,simu.prop_reject_null)


# utility function
def utilityCRRA(x,gamma):
        return x**(1-gamma)/(1-gamma)



