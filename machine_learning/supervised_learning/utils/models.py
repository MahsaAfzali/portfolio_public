import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def grid_search(
    models: dict,
    parameter_grid: list[dict],
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    number_cross_val_fold: int = 5,
    scoring_method: str = "accuracy",
    verbose: int = 1,
):
    # Set up a pipeline
    pipeline = Pipeline(
        [
            (
                "model",
                # model placeholder
                list(models.values())[0],
            )  
        ]
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        cv=number_cross_val_fold,
        scoring=scoring_method,
        verbose=verbose,
    )

    # fit the grid search
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation {scoring_method} score: {grid_search.best_score_}")

    return grid_search
