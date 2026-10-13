"""
Support Vector Machine from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - standardize_features
import numpy as np

def standardize_features(x):
    # TODO: rescale each column of x to have mean 0 and std 1 (leave zero-std columns alone).
    me = np.mean(x,axis=0)
    st = np.std(x,axis=0)
    st[st==0] = 1.0
    return (x-me)/st

# Step 2 - initialize_parameters
import numpy as np

def initialize_parameters(n_features):
    """Return a dict with 'w' of shape (n_features,) and scalar 'b'."""
    # TODO: create starting weights and bias for a linear SVM
    w=np.zeros(n_features)
    b= 0.0
    return {'w':w,'b':b}

# Step 3 - compute_scores
import numpy as np

def compute_scores(x, params):
    """Return raw linear scores x @ w + b, shape (n_samples,)."""
    # TODO: score each example as a linear function of the current weights and bias.
    return x@params['w']+params['b']

# Step 4 - predict_from_scores
import numpy as np

def predict_from_scores(scores):
    # TODO: convert a 1-D array of raw scores into +1 / -1 class predictions.
    return np.where(scores>=0,1,-1)

# Step 5 - hinge_loss_example
def hinge_loss_example(score, y):
    # TODO: return the hinge loss for a single example with raw score `score` and label y in {-1, +1}.
    return np.maximum(0.0,1-y*score)

# Step 6 - svm_objective
def svm_objective(x, y, params, reg_lambda):
    # TODO: return mean hinge loss over the dataset plus reg_lambda * (w dot w)
    w = params['w']

    # 步骤1：计算所有样本得分 s = X @ w + b
    scores = compute_scores(x,params)

    # 步骤2：向量化批量铰链损失
    hinge_all = hinge_loss_example(scores,y)

    # 步骤3：铰链损失取均值，除以样本数n
    hinge_mean = np.mean(hinge_all)

    # 步骤4：L2正则项，只惩罚w，b不加！ w·w = ||w||²
    l2_penalty = reg_lambda * np.dot(w, w)

    # 总目标
    J = hinge_mean + l2_penalty
    return float(J)

# Step 7 - compute_gradients (not yet solved)
# TODO: implement

# Step 8 - apply_update (not yet solved)
# TODO: implement

# Step 9 - train_svm (not yet solved)
# TODO: implement

# Step 10 - predict_labels (not yet solved)
# TODO: implement

# Step 11 - accuracy_score (not yet solved)
# TODO: implement

