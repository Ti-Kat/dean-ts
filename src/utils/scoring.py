from sklearn.metrics import roc_auc_score as auc


def compute_auc_score(y_score, y_true, print_result=False):
    """ Calculate auc score of a single model
    """
    auc_score = auc(y_true, y_score)
    if print_result:
        print(f'Reached AUC score of {auc_score}')
    return auc_score
