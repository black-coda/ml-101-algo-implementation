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
    import numpy as np

    # Add a bias term (intercept) to the input data
    X_b = np.c_[np.ones((X.shape[0], 1)), X]

    # Compute the parameters using the normal equation
    theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    return theta