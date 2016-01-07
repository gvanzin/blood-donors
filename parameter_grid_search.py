import pandas as pd
import graphlab as gl
import numpy as np
import matplotlib.pyplot as plt
from data_prep import modeling_prep
from evaluator import *
from sklearn.metrics import roc_auc_score
import graphlab as gl
'''these functions grid search for the best parameters across three models using 5 fold cross validation'''
def logistic_grid(train_set):
      logisticparams = dict([('target', 'target'),
                   ('class_weights', 'auto'),
                   ('convergence_threshold', [0.001,0.01, 0.1]),
                    ('feature_rescaling', 1),
                   ('l1_penalty', [0.0,0.001,0.01,0.1]),
                   ('l2_penalty', [0.001,0.01,0.1]),
                   ('lbfgs_memory_level', 11),
                   ('max_iterations', 20),
                   ('solver', 'auto'),
                   ('step_size', 1.0)])

      folds = gl.cross_validation.KFold(train_set, 5)

      #logistic regression grid search
      logistic_grid_job = gl.grid_search.create(folds,
                                    gl.logistic_classifier.create,
                                    logisticparams, evaluator=evaluate_auc)

      logistic_auc_results = logistic_grid_job.get_results()
      logistic_auc_results.sort('mean_validation_auc', ascending=False)
      logictic_best_params=logistic_auc_results.get_best_params('mean_vaidation_auc',ascending=False)
      return logictic_best_params

def random_forest_grid(train_set):
      random_params = dict([('target', 'target'),
                   ('column_subsample', '1.0'),
                   ('max_depth', [4,6,8]),
                    
                   ('min_child_weight', [0.1,0.5,1.0,2]),
                   ('min_loss_reduction', [0.0,1,2,3,4]),
                   ('num_trees', 100),
                   ('row_subsample', 0.8),
                   ('class_weights','auto'),
                   ('step_size', [0.5, 1.0, 2.0])])

      folds = gl.cross_validation.KFold(train_set, 5)
      #random forest  grid search
      random_grid_job = gl.grid_search.create(folds,
                                    gl.random_forest_classifier.create,
                                    random_params, evaluator=evaluate_auc)


      random_auc_results = random_grid_job.get_results()
      random_auc_results.sort('mean_validation_auc', ascending=False)
      random_best_params=random_auc_results.get_best_params()
      return random_best_params




def gradient_boosted_grid(train_set):
      boostedparams = {'target': 'target',
          'class_weights': 'auto',
          'min_child_weight': [1, 2, 4, 8, 16],
          'min_loss_reduction': [0, 1, 10],
           'row_subsample': [1, .9],
                 'column_subsample': [1, .9, .8],
            'max_depth': [4,6,8,10],
                 'max_iterations':100,
                 'step_size': [0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.25, 0.5]
         }



      folds = gl.cross_validation.KFold(train_set, 5)
      boosted_job_iterations = gl.grid_search.create(folds,
                            gl.boosted_trees_classifier.create,
                            boostedparams,
                            evaluator=evaluate_auc
                           )

      boosted_results = boosted_job_iterations.get_results()
      boosted_results.sort('mean_validation_auc', ascending=False)
      boosted_best_params=boosted_results.get_best_params()
      return boosted_best_params



def auc_score(model, test):
    target = model.get('target')
    preds = model.predict(test, output_type='class')
    return roc_auc_score(np.asarray(test[target]), np.asarray(preds))

def evaluate_auc(model, train, test):
    return {'train_auc': auc_score(model, train), 
         'validation_auc': auc_score(model, test)}



def main():

      '''this function creates the appropriate train/test datasets for modeling
      and runs the model.'''
      
      #data = load_data()
      train_set, test_set = modeling_prep()
      #model = random_forest(X_train,X,y_train,y)
      logistic_params=logistic_grid(train_set)
      random_params=random_forest_grid(train_set)
      gradient_params=gradient_boosted_grid(train_set)
      return logistic_params,random_params,gradient_params

      

if __name__ == '__main__':
      log_params,rf_params,gb_params=main()


