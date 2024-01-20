import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV


def create_random_search_grid_for_random_forest():
    """
    Creates a random grid for random forest search to find the best possible hyperparameters
    """
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=50, stop=250, num=5)]

    # Number of features to consider at every split
    max_features = ['log2', 'sqrt']

    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)

    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]

    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]

    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    print(random_grid)
    return random_grid


def train_random_forest_for_best_parameters(x_train, y_train):
    rf_reg = RandomForestRegressor()
    rf_random = RandomizedSearchCV(
        estimator=rf_reg,
        param_distributions=create_random_search_grid_for_random_forest(),
        n_iter=100,
        cv=3,
        random_state=42,
        n_jobs=-1,
    )

    rf_random.fit(x_train, y_train)

    print(rf_random.best_params_)