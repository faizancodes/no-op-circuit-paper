    True coefficients: [20.4923...  34.1698...]
    >>> print("Huber coefficients:", huber.coef_)
    Huber coefficients: [17.7906... 31.0106...]
    >>> print("Linear Regression coefficients:", linear.coef_)
    Linear Regression coefficients: [-1.9221...  7.0226...]

    References
    ----------
    .. [1] Peter J. Huber, Elvezio M. Ronchetti, Robust Statistics
           Concomitant scale estimates, pg 172
    .. [2] Art B. Owen (2006), A robust hybrid of lasso and ridge regression.
           https://statweb.stanford.edu/~owen/reports/hhu.pdf
    """

    def __init__(self, epsilon=1.35, max_iter=100, alpha=0.0001,
                 warm_start=False, fit_intercept=True, tol=1e-05):
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.alpha = alpha
        self.warm_start = warm_start
        self.fit_intercept = fit_intercept
        self.tol = tol

    def fit(self, X, y, sample_weight=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        sample_weight : array-like, shape (n_samples,)
            Weight given to each sample.

        Returns
        -------
        self : object
        """
        X, y = check_X_y(
            X, y, copy=False, accept_sparse=['csr'], y_numeric=True)
        if sample_weight is not None:
            sample_weight = np.array(sample_weight)
            check_consistent_length(y, sample_weight)
        else:
            sample_weight = np.ones_like(y)

        if self.epsilon < 1.0:
            raise ValueError(
                "epsilon should be greater than or equal to 1.0, got %f"
                % self.epsilon)

        if self.warm_start and hasattr(self, 'coef_'):
            parameters = np.concatenate(
                (self.coef_, [self.intercept_, self.scale_]))
        else:
            if self.fit_intercept:
                parameters = np.zeros(X.shape[1] + 2)
            else:
                parameters = np.zeros(X.shape[1] + 1)
            # Make sure to initialize the scale parameter to a strictly
            # positive value:
            parameters[-1] = 1

        # Sigma or the scale factor should be non-negative.
        # Setting it to be zero might cause undefined bounds hence we set it
        # to a value close to zero.
        bounds = np.tile([-np.inf, np.inf], (parameters.shape[0], 1))
        bounds[-1][0] = np.finfo(np.float64).eps * 10

        parameters, f, dict_ = optimize.fmin_l_bfgs_b(
            _huber_loss_and_gradient, parameters,
            args=(X, y, self.epsilon, self.alpha, sample_weight),
            maxiter=self.max_iter, pgtol=self.tol, bounds=bounds,
            iprint=0)
        if dict_['warnflag'] == 2:
            raise ValueError("HuberRegressor convergence failed:"
                             " l-BFGS-b solver terminated with %s"
                             % dict_['task'].decode('ascii'))
        # In scipy <= 1.0.0, nit may exceed maxiter.
        # See https://github.com/scipy/scipy/issues/7854.
        self.n_iter_ = min(dict_['nit'], self.max_iter)
        self.scale_ = parameters[-1]
        if self.fit_intercept:
