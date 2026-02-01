import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta = None
        self.cost = []

    def _add_intercept(self, X):
        return np.c_[np.ones((X.shape[0], 1)), X]

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _compute_log_loss(self, X, y):
        m = X.shape[0]
        z = X @ self.theta
        p = self._sigmoid(z)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -(1 / m) * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    def _compute_gradient(self, X, y):
        m = X.shape[0]
        p = self._sigmoid(X.dot(self.theta))
        error = p - y
        return (1 / m) * X.T.dot(error)

    def fit(self, X, y):
        X = self._add_intercept(X)
        self.theta = np.zeros(X.shape[1])

        for _ in range(self.n_iterations):
            gradients = self._compute_gradient(X, y)
            self.theta -= self.learning_rate * gradients
            self.cost.append(self._compute_log_loss(X, y))

        return self.theta

    def predict_proba(self, X):
        X = self._add_intercept(X)
        return self._sigmoid(X.dot(self.theta))

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    import numpy as np

    X, y = make_classification(
        n_samples=300,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42,
    )

    model = LogisticRegression(learning_rate=0.1, n_iterations=2000)
    model.fit(X, y)
    y_pred_custom = model.predict(X)
    custom_accuracy = np.mean(y_pred_custom == y)
    print("Custom Logistic Regression accuracy:", custom_accuracy)

    from sklearn.linear_model import LogisticRegression as SkLogReg

    sk_model = SkLogReg(penalty="none", solver="lbfgs", max_iter=2000)
    sk_model.fit(X, y)
    y_pred_sk = sk_model.predict(X)

    sk_accuracy = np.mean(y_pred_sk == y)
    print("Sklearn Logistic Regression accuracy:", sk_accuracy)

    theta_custom = model.theta
    theta_sk = np.hstack([sk_model.intercept_, sk_model.coef_.flatten()])
