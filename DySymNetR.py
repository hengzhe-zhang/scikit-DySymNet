from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sympy import parse_expr, preorder_traversal

from .SymbolicRegression import DySymNetRegressor
from .scripts.params import Params

config = Params()
est = DySymNetRegressor(config=config, func_name="")


def complexity(est):
    return len(list(preorder_traversal(parse_expr(est.model()))))


def model(est):
    return str(est.model())


if __name__ == "__main__":
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train)
    X_test = x_scaler.transform(X_test)
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    est.fit(X_train, y_train)
    print(r2_score(y_test, est.predict(X_test)))
    print(est.model())
    print("Complexity: ", complexity(est))
