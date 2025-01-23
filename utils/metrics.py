import numpy as np

def rmse(pred, true):
    rmse = np.sqrt(np.mean((pred - true) ** 2))
    return rmse

def mse(pred, true):
    mse = np.mean((pred - true) ** 2)
    return mse

def MAE(pred, true):
    return np.mean(np.abs(true - pred))

def metric(preds, trues):
    pred_cat = np.concatenate(preds, axis=0)
    true_cat = np.concatenate(trues, axis=0)
    # print(list(pred_cat - true_cat))
    res = {}
    res['rmse'] = rmse(pred_cat, true_cat)
    res['mse'] = mse(pred_cat, true_cat)
    res['mae'] = MAE(pred_cat, true_cat)
    return res