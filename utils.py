
import numpy as np
import pandas as pd
import scipy.stats as st

# -----------------------------
#    Some cognitive functions
# -----------------------------
# CRRA utility function
# Note riskCoef ≠ 1. I suggest using 2 as the initial value.
def uCRRA(x,coef):
    try:
        if isinstance(x[0][0],np.ndarray):
            x = np.apply_along_axis(lambda x: x.tolist(), axis=1, arr=x)
    except:
        pass
    
    return np.array(x)**coef

# exponential discount function
def discountExp(t,coef):
    return coef**np.array(t)

# probability weighting function
def probWeight(p,coef=1):
    try:
        if isinstance(p[0][0],np.ndarray):
            p = np.apply_along_axis(lambda x: x.tolist(), axis=1, arr=p)
    except:
        pass

    return np.exp(-(-np.log(np.array(p)))**coef)
    

# choice probability under the logit model
def logitProb(params,x1,x2,p1=1,p2=1,t1=0,t2=0):
        
        '''
        Each choice involves two options
            option 1: (x1,t1,p1)
            option 2: (x2,t2,p2)
        x is the amounts, t is the reward delivery times, p is the probabilities

        u_diff = u(x1,t1,p1) - u(x2,t2,p2)

        choice probability for option 1: 
            P(A) = sigmoid( u_diff / temp )
        where temp is the temperature parameter
    '''

        if 'probW' not in params.keys():
            params['probW'] = 1
        
        u_1 = uCRRA(x1,params['riskCoef']) * probWeight(p1,params['probW']) * discountExp(t1,params['delta'])
        u_2 = uCRRA(x2,params['riskCoef']) * probWeight(p2,params['probW']) * discountExp(t2,params['delta'])

        if len(u_1.shape) > 2:
            u_1 = u_1.sum(axis=2)
        
        if len(u_2.shape) > 2:
            u_2 = u_2.sum(axis=2)

        u_diff = (u_1 - u_2)

        prob = 1/(1+ np.exp(-u_diff / params['temp']))

        return prob


def logitProb_expanded(params,x1,x2,p1=1,p2=1,t1=0,t2=0):
        
    '''
    This is function is the same as logitProb(.) but the params are random parameters. 
    '''

    var_name_list = ['x1', 'x2', 'p1', 'p2', 't1', 't2']
    coef_name_list = list(params.keys())
    vars = {}
    coefs_1 = {}
    coefs_2 = {}

    # n_draws: Compute the number of draws for each individual
    # n_indiv : Compute the (max) number of indivdiduals
    # n_q : Compute the (max) number of questions answered by each individual
    # n_state_1, n_state_2 : Compute the number of states for each option 
    n_draws = params['temp'].shape[1]

    n_indiv = 1
    n_q = 1
    for var_name in var_name_list:
        var = eval(var_name)
        if not np.isscalar(var):  
            n_indiv = max(n_indiv, len(var))
            if len(var.shape) > 1:
                n_q = max(n_q,var.shape[1])

    # Check whether an option contains multiple states.
    try:
        n_state_1 = len(x1[0][0])
    except:
        n_state_1 = 1

    try:
        n_state_2 = len(x2[0][0])
    except:
        n_state_2 = 1
    
    
    # If the option has only a single state, each parameter and variable should match (n_q, n_indiv, n_draws)
    # If the option has multiple states, each parameter and variable should match (n_q, n_indiv, n_state, n_draws)
    # Check whether a variable is scalar. If so, repeat it multiple times to match the given shape.
    for coef in coef_name_list:
        coefs_1[coef] = np.tile(params[coef], (n_q, 1, 1))
        coefs_2[coef] = np.tile(params[coef], (n_q, 1, 1))

        if n_state_1 > 1:
            coefs_1[coef] = np.expand_dims(coefs_1[coef], axis=2)
        
        if n_state_2 > 1:
            coefs_2[coef] = np.expand_dims(coefs_2[coef], axis=2)
    

    # Check whether a variable is scalar. If so, repeat it multiple times to match the given shape.
    for var_name in ['x1','p1','t1']:
        var = eval(var_name)
        if np.isscalar(var): 
            if n_state_1 == 1:
                var = np.full((n_indiv, n_q), var)
            else:
                var = np.full((n_state_1, n_indiv, n_q ), var)
    
        vars[var_name] = var.T[..., np.newaxis]
        
    
    for var_name in ['x2','p2','t2']:
        var = eval(var_name)
        if np.isscalar(var): 
            if n_state_2 == 1:
                var = np.full((n_indiv, n_q), var)
            else:
                var = np.full((n_state_2, n_indiv, n_q ), var)
    
        vars[var_name] = var.T[..., np.newaxis]
    

    u_1 = uCRRA(vars['x1'],coefs_1['riskCoef']) * probWeight(vars['p1'],coefs_1['probW']) * discountExp(vars['t1'],coefs_1['delta'])
    u_2 = uCRRA(vars['x2'],coefs_2['riskCoef']) * probWeight(vars['p2'],coefs_2['probW']) * discountExp(vars['t2'],coefs_2['delta'])

    if len(u_1.shape) > 3:
        u_1 = u_1.sum(axis=2)
    
    if len(u_2.shape) > 3:
        u_2 = u_2.sum(axis=2)
    
    u_diff = u_1 - u_2

    temp = np.tile(params['temp'], (n_q, 1, 1))

    prob = 1/(1+ np.exp(-u_diff /temp))

    return prob


# -----------------------------
#    Encoding Choice Path 
# -----------------------------

def from_choice_to_encode(n, bits):
    """
    Given a number n and a binary list `bits`, continuously perform binary division
    on the interval [1, n]. At each step:
      - If the bit is 0, select the lower half of the current interval.
      - If the bit is 1, select the upper half of the current interval.
      
    Parameters:
      n: An integer representing the upper bound of the initial interval [1, n].
      bits: A list of 0s and 1s. Each bit determines whether to choose the lower or upper half.
      
    Returns:
      If the interval is reduced to a single number, returns that number;
      otherwise, returns a tuple (lower, upper) representing the final interval.
    """
    lower, upper = 1, n
    for b in bits:
        # Calculate the midpoint using integer division.
        # For an even count of numbers, this will split the interval equally.
        # For example, for [1, 32]: mid = (1 + 32) // 2 = 16,
        # so the lower half is [1, 16] and the upper half is [17, 32].
        mid = (lower + upper) // 2
        if b == 0:
            # Choose the lower half: [lower, mid]
            upper = mid
        elif b == 1:
            # Choose the upper half: [mid+1, upper]
            lower = mid + 1
        else:
            raise ValueError("Elements in the bits list must be either 0 or 1")
    
    # If the interval is reduced to a single number, return that number;
    # otherwise, return the interval as a tuple (lower, upper)
    if lower == upper:
        return lower
    else:
        return (lower, upper)

# -----------------------------
#    Proportion Test
# -----------------------------
class proportionTest:

    def __init__(self,tab_count):
        self.tab_count = tab_count
    
    # Run the test for a node
    @staticmethod    
    def single_prop_test(node_count):
        # Number of observations in each node
        # hi: higher level for option B (the variable option)
        # lo: lower level for option B
        n_lo = node_count[0] + node_count[1]
        n_hi = node_count[2] + node_count[3]

        # Number of observations choosing option A (the fixed option)
        q_lo = node_count[1]
        q_hi = node_count[3]

        # p(g): proportion of choosing option A in each group, where g is selected from {lo,hi}
        # p(0) denotes the parent node
        # Null Hypothesis: p(lo) >= p(hi)
        # Fisher's & Barnard's exact test
        matrix_lo_hi = np.array([[q_lo, n_lo - q_lo], [q_hi, n_hi - q_hi]])
        p_fisher = st.fisher_exact(matrix_lo_hi, alternative='less').pvalue
        p_barnard = st.barnard_exact(matrix_lo_hi,alternative='less',pooled=False).pvalue

        # Create a new row
        new_row = {
            "n": n_lo + n_hi,
            "prop_lo": q_lo / n_lo if n_lo > 0 else np.nan,
            "prop_0": n_hi / (n_lo + n_hi) if (n_lo + n_hi) > 0 else np.nan,
            "prop_hi": q_hi / n_hi if n_hi > 0 else np.nan,
            "p_fisher": p_fisher,
            "p_barnard": p_barnard,
            "n_lo": n_lo,
            "n_hi": n_hi,
            "q_lo": q_lo,
            "q_hi": q_hi
        }

        return new_row

    # Write test results based on the count data 
    def run(self,hide=False):

        for i in range(1, len(self.tab_count)+1):
            if str(i) not in self.tab_count.index:
                self.tab_count[str(i)] = 0

        self.tab_count.index = self.tab_count.index.astype(int)
        self.tab_count = self.tab_count.sort_index()

        result_prop_test = pd.DataFrame({})

        max_layer = int(np.log2(len(self.tab_count)))-1

        for i in range(max_layer+1):

            new_table = [sum(self.tab_count[x:x + 2**i]) for x in range(0, len(self.tab_count), 2**i)]

            max_group = len(self.tab_count) // ((i + 1) * 4) - 1

            # Iterate over groups
            for g in range(int(max_group) + 1):
                # Select the current group of 4 elements
                node_count = new_table[4 * g:4 * g + 4]

                if hide == False:
                    print(f"Testing layer {max_layer - i}, group {g + 1}")
                
                try:
                    # Perform the prop test
                    new_row = self.single_prop_test(node_count)
                    
                    # Add layer and group information
                    new_row.update({"layer": max_layer - i, "group": g + 1})
                    
                    # Append the result as a new row
                    result_prop_test = pd.concat([result_prop_test, pd.DataFrame([new_row])], ignore_index=True)
                
                except Exception as e:
                    print(f"Error: {e}")
                    print(node_count)

        self.result = result_prop_test
        return result_prop_test

    def get_overall_pvalue(self,method='fisher'):
        prop_test_sort = sorted(self.result[[f'p_{method}']].values.flatten(),reverse= True)
        pvalue = min(np.array(prop_test_sort) * np.arange(1,len(prop_test_sort)+1))
        return pvalue

    @staticmethod
    def single_test_power(q_lo, q_hi, n_lo, n_hi, n_simu=1000, alpha = 0.5, adj_term = 1, method='fisher'):
        # power analysis
        H1_p = []
        for _ in range(n_simu):
            # simulate 2 × 2 contingency table
            cell_1_1 = np.random.binomial(q_lo, q_lo / n_lo)  
            cell_1_2 = q_lo - q_lo                     
            cell_2_1 = np.random.binomial(q_hi, q_hi / n_hi)  
            cell_2_2 = n_hi - q_hi                      
            
            # Obtain p-values assuming H1 (alternative hypothesis) is true
            simu_matrix = np.array([[cell_1_1, cell_1_2], [cell_2_1, cell_2_2]])

            if method=='fisher':
                H1_p_raw = st.fisher_exact(simu_matrix, alternative='greater').pvalue
            elif method =='barnard':
                H1_p_raw = st.barnard_exact(simu_matrix,alternative='greater',pooled=False).pvalue

            H1_p += [H1_p_raw]
        
        power = np.mean(np.array(H1_p) < alpha/adj_term)
        return power

    def get_power(self,method='fisher'):
        new_result = self.result.sort_values(f'p_{method}',ascending=False)
        power_list = []

        for r in range(len(new_result)):
            adj_term = r + 1
            row = new_result.iloc[r,:]
            _power = self.single_test_power(row['q_lo'], row['q_hi'], row['n_lo'], row['n_hi'], 
                            adj_term=adj_term, method=method)
            power_list += [_power]

        new_result[f'power_{method}'] = power_list
        self.result = new_result
        self.overall_power = max(power_list)
        return power_list

# -----------------------------
#    Regression Table
# -----------------------------
def get_star(p):
    if p > 0.05:
        return ''
    elif p > 0.01:
        return '$^{*}$'
    elif p > 0.005:
        return '$^{**}$'
    else:
        return '$^{***}$'



def fill_cell(var,reg_result,digit = 3,robust_se=False):

    cell = ""

    if var.split('_')[0] == 'b':
        try:
            if not robust_se:
                coef = round(reg_result.params[var.split('_')[1]],digit)
                cell_pvalue = reg_result.pvalues[var.split('_')[1]]
                cell = str(coef) + get_star(cell_pvalue)
            else:
                var_idx = np.where(np.array(reg_result.model.exog_names) == var.split('_')[1])[0][0]
                coef = round(reg_result.params[var_idx],digit)
                cell_pvalue = reg_result.pvalues[var_idx]
                cell = str(coef) + get_star(cell_pvalue)
        except:
            pass

    elif var.split('_')[0] == 'se':
        try:
            if not robust_se:
                bse = round(reg_result.bse[var.split('_')[1]],digit)
                cell = f'({bse})'
            else:
                var_idx = np.where(np.array(reg_result.model.exog_names) == var.split('_')[1])[0][0]
                bse = round(reg_result.bse[var_idx],digit)
                cell = f'({bse})'
        except:
            pass

    elif var == 'nobs':
        cell = round(reg_result.nobs)

    elif var in ['rsquared_adj','aic']:
        cell = round(getattr(reg_result,var),digit)
    
    return str(cell)



def add_border(input_string):

    # Replace '\toprule', '\midrule', '\bottomrule' with '\hline'
    output_string = input_string.replace('\\toprule', '\\hline').replace('\\midrule', '\\hline').replace('\\bottomrule', '\\hline')
    
    # Insert '\hline' before '\nobservations'
    # index = output_string.find('\nobservations')
    # output_string = output_string[:index] + '\\hline\n' + output_string[index:]

    return output_string


def make_table(input_df,output_path):
    with open(output_path,'w') as f:
        # tex_code = '\\documentclass[12px]{article} \n \\begin{document} \n' + input_df.to_latex() + '\n \end{document}'
        tex_code = input_df.to_latex()
        tex_code = add_border(tex_code)
        f.write(tex_code)
