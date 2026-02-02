import optuna
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score

def optimize_svr(X, y):
    def objective(trial):
        c = trial.suggest_float('C', 1e-1, 1e5, log=True)
        epsilon = trial.suggest_float('epsilon', 1e-3, 1e1, log=True)
        gamma = trial.suggest_float('gamma', 1e-5, 1e-1, log=True)
        
        svr = SVR(kernel='rbf', C=c, epsilon=epsilon, gamma=gamma)
        score = cross_val_score(svr, X, y, cv=3, scoring='neg_mean_squared_error').mean()
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50) # 時間に合わせて調整
    return study.best_params

def train_final_model(X, y, params):
    model = SVR(**params)
    model.fit(X, y)
    return model