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

# Step 7 - compute_gradients
import numpy as np

def compute_gradients(x, y, params, reg_lambda):
    """Return {'dw': ndarray shape (n_features,), 'db': float} = gradient of svm_objective."""
    # TODO: compute the gradient of the SVM objective wrt params['w'] and params['b'].
    """
    计算SVM子梯度 dw, db
    X: (n_samples, n_features)
    y: (n_samples,) ∈ {+1,-1}
    params: {'w':w, 'b':b}, w: (n_features,)
    reg_lambda: λ 正则系数
    return {'dw': np.ndarray, 'db': float}
    """
    w = params['w']
    b = params['b']
    n = x.shape[0]

    # 1. 计算所有样本得分
    scores = x @ w + b

    # 2. margin向量 m_i = 1 - y_i * s_i
    margins = 1.0 - y * scores

    # 3. 激活样本掩码 m_i > 0
    mask = (margins > 0)   # boolean数组 shape(n_samples,)

    # 4. 计算dw
    sum_yixi = x.T @ (mask * y)   # ∑_{m_i>0} y_i x_i
    dw = (-1.0 / n) * sum_yixi + 2 * reg_lambda * w

    # 5. 计算db
    sum_yi = np.sum(mask * y)     # ∑_{m_i>0} y_i
    db = (-1.0 / n) * sum_yi

    return {"dw": dw, "db": float(db)}

# Step 8 - apply_update
def apply_update(params, grads, learning_rate):
    # TODO: return a new params dict after one gradient-descent step on 'w' and 'b'.
    w_old = params['w']
    b_old = params['b']

    dw = grads['dw']
    db = grads['db']

    # 梯度下降更新：w = w - η * dw
    w_new = w_old - learning_rate * dw
    b_new = float(b_old - learning_rate * db)

    # 返回全新字典，不改动外部输入对象
    return {
        "w": w_new,
        "b": b_new
    }

# Step 9 - train_svm (not yet solved)
# TODO: implement

# Step 10 - predict_labels (not yet solved)
# TODO: implement

# Step 11 - accuracy_score (not yet solved)
# TODO: implement

