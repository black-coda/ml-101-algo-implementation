from typing import Optional, Union
import numpy as np
import pandas as pd


def normal_equation(X, y):
    """
    Compute the parameters of a linear regression model using the normal equation.

    Parameters:
    X : array-like, shape (n_samples, n_features)
        The input data.
    y : array-like, shape (n_samples,)
        The target values.

    Returns:
    theta : array, shape (n_features,)
        The parameters of the linear regression model.
    """

    # Add a bias term (intercept) to the input data
    X_b = np.c_[np.ones((X.shape[0], 1)), X]

    # Compute the parameters using the normal equation
    theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    # the theta can also be computed using the @:
    # theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

    # for case the matrix is non-invertible, we can use the pseudo-inverse:
    # this introduces the SVD (singular value decomposition) under the hood
    # the LinearRegression from sklearn uses this approach under the hood

    # x
    # {(N,), (N, K)} ndarray

    #     Least-squares solution. If b is two-dimensional, the solutions are in the K columns of x.
    # residuals
    # {(1,), (K,), (0,)} ndarray

    #     Sums of squared residuals: Squared Euclidean 2-norm for each column in b - a @ x. If the rank of a is < N or M <= N, this is an empty array. If b is 1-dimensional, this is a (1,) shape array. Otherwise the shape is (K,).
    # rank
    # int

    #     Rank of matrix a.
    # s
    # (min(M, N),) ndarray

    #     Singular values of a.

    theta_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=None)

    return theta


class GradientDescent:
    """
    A simple implementation of gradient descent for linear regression.
    
    Supports both batch and stochastic gradient descent with automatic
    bias/intercept handling to match the normal equation approach.
    
    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size for gradient descent updates
    n_iterations : int, default=1000
        Maximum number of iterations (batch) or epochs (stochastic)
    tolerance : float, default=1e-5
        Convergence threshold for MSE change between iterations
    fit_intercept : bool, default=True
        Whether to add a bias/intercept term (column of ones)
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        tolerance: float = 1e-5,
        fit_intercept: bool = True,
    ):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.tolerance = tolerance
        self.fit_intercept = fit_intercept
        
        # These will be set during fit()
        self.X = None
        self.y = None
        self.theta = None
        self.mse_history = []

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """
        Add bias column to feature matrix if fit_intercept is True.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input features
            
        Returns
        -------
        X_b : ndarray of shape (n_samples, n_features + 1) or (n_samples, n_features)
            Features with bias column prepended if fit_intercept=True
        """
        if self.fit_intercept:
            return np.c_[np.ones((X.shape[0], 1)), X]
        return X

    def compute_mse_vectorized(self):
        """
        Compute the Mean Squared Error (MSE) using vectorized operations.
        
        MSE is the cost function for linear regression. This version uses
        matrix multiplication for the quadratic form.
        
        Returns
        -------
        mse : float
            The mean squared error
        """
        n = self.y.shape[0]
        predicted_y = self.X @ self.theta
        differences = predicted_y - self.y
        mse = 1 / n * (differences.T @ differences)
        return mse

    def compute_mse(self):
        """
        Compute the Mean Squared Error (MSE) using element-wise operations.
        
        This version is more readable and numerically equivalent to
        compute_mse_vectorized().
        
        Returns
        -------
        mse : float
            The mean squared error
        """
        predicted_y = self.X @ self.theta
        differences = predicted_y - self.y
        return np.mean(np.square(differences))

    def _compute_gradients(self):
        """
        Compute the gradient of MSE with respect to theta (batch version).
        
        Uses the analytical gradient: ∇MSE = (2/m) * X^T * (X*theta - y)
        
        Returns
        -------
        gradients : ndarray of shape (n_features,)
            Gradient vector for all parameters
        """
        m = self.y.shape[0]
        predicted_y = self.X @ self.theta
        differences = predicted_y - self.y
        return 2 / m * (self.X.T @ differences)

    def fit(self, X: np.ndarray, y: np.ndarray, method: str = 'batch') -> np.ndarray:
        """
        Fit linear regression model using gradient descent.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training feature matrix
        y : ndarray of shape (n_samples,)
            Training target values
        method : str, default='batch'
            Optimization method: 'batch' or 'stochastic'
            
        Returns
        -------
        theta : ndarray
            Fitted parameters (including intercept if fit_intercept=True)
        """
        # Input validation
        if X is None or y is None:
            raise ValueError("X and y cannot be None")
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y shape mismatch: X has {X.shape[0]} samples, "
                f"y has {y.shape[0]} samples"
            )
        
        # Store data with intercept if needed
        self.X = self._add_intercept(X)
        self.y = y
        self.mse_history = []
        
        # Choose optimization method
        if method == 'batch':
            return self.batch_gradient_descent()
        elif method == 'stochastic':
            return self.stochastic_gradient_descent(epochs=self.n_iter)
        else:
            raise ValueError(
                f"Unknown method '{method}'. Use 'batch' or 'stochastic'"
            )

    def batch_gradient_descent(self) -> np.ndarray:
        """
        Fit model using batch gradient descent.
        
        Updates parameters using the entire dataset at each iteration.
        Converges when change in MSE falls below tolerance.
        
        Returns
        -------
        theta : ndarray
            Fitted parameters
        """
        n_samples, n_features = self.X.shape
        self.theta = np.zeros(n_features)
        previous_mse = None
        
        for i in range(self.n_iter):
            # compute gradients and update theta(weights)
            gradients = self._compute_gradients()
            self.theta -= self.lr * gradients

            # compute the mse to check for convergence
            mse = self.compute_mse()
            self.mse_history.append(mse)

            if previous_mse is not None and abs(previous_mse - mse) < self.tolerance:
                print(f"Converged after {i+1} iterations.")
                break
            previous_mse = mse

        return self.theta

    def learning_schedule(self, t: int) -> float:
        """
        Compute decaying learning rate for stochastic gradient descent.
        
        Uses the formula: eta = t0 / (t + t1)
        
        Parameters
        ----------
        t : int
            Current iteration number (total updates so far)
            
        Returns
        -------
        eta : float
            Learning rate for this iteration
        """
        t0, t1 = 5, 50  # learning schedule hyperparameters
        return t0 / (t + t1)

    def stochastic_gradient_descent(self, epochs: int = 50) -> np.ndarray:
        """
        Fit model using stochastic gradient descent.
        
        Updates parameters using one random sample at a time with a
        decaying learning rate. More efficient for large datasets.
        
        Parameters
        ----------
        epochs : int, default=50
            Number of passes through the entire dataset
            
        Returns
        -------
        theta : ndarray
            Fitted parameters
        """
        n_samples, n_features = self.X.shape
        self.theta = np.zeros(n_features)
        previous_mse = None

        for epoch in range(epochs):
            # Shuffle indices for each epoch (better convergence)
            indices = np.random.permutation(n_samples)
            
            for idx in indices:
                # Pick a random sample (using shuffled indices)
                xi = self.X[idx:idx+1]  # Keep 2D shape
                yi = self.y[idx:idx+1]

                # Compute gradients for single sample and update theta(weights)
                # Gradient for single sample: 2 * X^T * (X*theta - y)
                gradients = 2 * xi.T @ (xi @ self.theta - yi)

                # Update learning rate based on total number of updates so far
                # t = epoch * n_samples + i = total number of updates
                eta = self.learning_schedule(epoch * n_samples + idx)
                self.theta -= eta * gradients.ravel()

            # Check convergence once per epoch (not per sample for efficiency)
            mse = self.compute_mse()
            self.mse_history.append(mse)

            if previous_mse is not None and abs(previous_mse - mse) < self.tolerance:
                print(f"Converged after {epoch+1} epochs.")
                break
            previous_mse = mse

        return self.theta

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted model.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input features (without bias column)
            
        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Predicted target values
        """
        if self.theta is None:
            raise ValueError(
                "Model must be fitted before making predictions. "
                "Call fit() first."
            )
        
        # Add intercept if model was trained with it
        X_b = self._add_intercept(X)
        return X_b @ self.theta

    def get_coefficients(self) -> dict:
        """
        Get the fitted coefficients in a readable format.
        
        Returns
        -------
        coef_dict : dict
            Dictionary with 'intercept' and 'coefficients' keys
        """
        if self.theta is None:
            raise ValueError("Model must be fitted first")
        
        if self.fit_intercept:
            return {
                'intercept': self.theta[0],
                'coefficients': self.theta[1:]
            }
        else:
            return {
                'intercept': 0.0,
                'coefficients': self.theta
            }