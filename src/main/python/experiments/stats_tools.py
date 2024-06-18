import numpy as np
from scipy import stats as st


def p_val_matrix_to_vec(matrix):

    ret_vals = []
    for r in range(matrix.shape[0]):
        for c in range(r + 1,matrix.shape[1]):
            ret_vals.append(matrix[r,c])

    return ret_vals

def p_val_vec_to_matrix(p_vec, num_rows,symmetrize=False):
    p_val_matrix = np.zeros( (num_rows, num_rows) )

    cnt = 0

    for r in range(p_val_matrix.shape[0]):
        for c in range(r,p_val_matrix.shape[1]):
            if r == c:
                p_val_matrix[r,c] = 1
                continue
            
            p_val_matrix[r,c] = p_vec[cnt]
            if symmetrize:
                p_val_matrix[c,r] = p_vec[cnt]
            cnt +=1

    return p_val_matrix

def reorder_names(li, names):
    """
    Function to reorder method names in experimental scripts.

    Arguments:
    ----------
    li -- list of values (names)
    names -- prefered order of method names

    Returns:
    --------
    indices, names
    """
    di = {name:idx for idx, name in enumerate(li)}
    
    indices_list = []
    names_list = []
    
    for name in names:
        if name in di:
            indices_list.append( di[name] )
            names_list.append( name )

    return indices_list, names_list

def chi_homogenity(contingency_table):
    """
    Function to calculate ch-square homogenity test on contingency table

    Arguments:
    ----------
    contingency_table -- contingency table to perform test 
        rows -- method
        columns -- values

    Returns:
    --------
    chi_square, p_value

    """
    # Calculate totals
    row_totals = np.array([np.sum(contingency_table, axis=1)])
    col_totals = np.array([np.sum(contingency_table, axis=0)])
    n = np.sum(contingency_table)
    # Calculate the expected observations
    expected = np.dot(row_totals.T, col_totals) / n + 1e-6

    if np.allclose(contingency_table, expected):
        return 0,1

    chisq, p_value = st.chisquare(contingency_table + 1e-6, expected)
    # Sum the answers
    chisq = np.sum(chisq)
    # Degrees of freedom
    rows = contingency_table.shape[0]
    cols = contingency_table.shape[1]
    dof = (rows - 1) * (cols - 1)
    # Convert chi-square test statistic to p-value
    p_value = 1 - st.chi2.cdf(chisq, dof)
    
    return chisq, p_value