
import numpy as np
import pandas as pd
import scipy.stats as st

# -----------------------------
#    Some useful functions
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
#    Proportion Test
# -----------------------------
# Run the test for a node
def run_prop_test(node_count):
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
    # Fisher's exact test
    matrix_lo_hi = [[q_lo, n_lo - q_lo], [q_hi, n_hi - q_hi]]
    t,p = st.fisher_exact(matrix_lo_hi, alternative='less')

    # Create a new row
    new_row = {
        "n": n_lo + n_hi,
        "prop_lo": q_lo / n_lo if n_lo > 0 else np.nan,
        "prop_0": (q_lo + q_hi) / (n_lo + n_hi) if (n_lo + n_hi) > 0 else np.nan,
        "prop_hi": q_hi / n_hi if n_hi > 0 else np.nan,
        "p_lo_hi": p,
    }

    return new_row

# Write test results based on the count data 
def prop_test(tab_count,hide=False):

    for i in range(1, len(tab_count)+1):
        if str(i) not in tab_count.index:
            tab_count[str(i)] = 0

    tab_count.index = tab_count.index.astype(int)
    tab_count = tab_count.sort_index()

    result_prop_test = pd.DataFrame({})

    max_layer = int(np.log2(len(tab_count)))-1

    for i in range(max_layer+1):

        new_table = [sum(tab_count[x:x + 2**i]) for x in range(0, len(tab_count), 2**i)]

        max_group = len(tab_count) // ((i + 1) * 4) - 1

        # Iterate over groups
        for g in range(int(max_group) + 1):
            # Select the current group of 4 elements
            node_count = new_table[4 * g:4 * g + 4]

            if hide == False:
                print(f"Testing layer {max_layer - i}, group {g + 1}")
            
            try:
                # Perform the prop test
                new_row = run_prop_test(node_count)
                
                # Add layer and group information
                new_row.update({"layer": max_layer - i, "group": g + 1})
                
                # Append the result as a new row
                result_prop_test = pd.concat([result_prop_test, pd.DataFrame([new_row])], ignore_index=True)
            
            except Exception as e:
                print(f"Error: {e}")
                print(node_count)
    
    return result_prop_test


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



def fill_cell(var,reg_result,digit = 3):

    cell = ""

    if var.split('_')[0] == 'b':
        try:
            coef = round(reg_result.params[var.split('_')[1]],digit)
            cell_pvalue = reg_result.pvalues[var.split('_')[1]]
            cell = str(coef) + get_star(cell_pvalue)
        except:
            pass

    elif var.split('_')[0] == 'se':
        try:
            bse = round(reg_result.bse[var.split('_')[1]],digit)
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
