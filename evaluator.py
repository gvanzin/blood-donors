from sklearn.metrics import roc_auc_score
def auc_score(model, test):
    target = model.get('target')
    preds = model.predict(test, output_type='class')
    return roc_auc_score(np.asarray(test[target]), np.asarray(preds))

def evaluate_auc(model, train, test):
    return {'train_auc': auc_score(model, train), 
         'validation_auc': auc_score(model, test)}