    estimator : estimator object, default=BayesianRidge()
        The estimator to use at each step of the round-robin imputation.
        If `sample_posterior=True`, the estimator must support
        `return_std` in its `predict` method.

    missing_values : int or np.nan, default=np.nan
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. For pandas' dataframes with
        nullable integer dtypes with missing values, `missing_values`
        should be set to `np.nan`, since `pd.NA` will be converted to `np.nan`.

    sample_posterior : bool, default=False
        Whether to sample from the (Gaussian) predictive posterior of the
        fitted estimator for each imputation. Estimator must support
        `return_std` in its `predict` method if set to `True`. Set to
        `True` if using `IterativeImputer` for multiple imputations.

    max_iter : int, default=10
        Maximum number of imputation rounds to perform before returning the
        imputations computed during the final round. A round is a single
        imputation of each feature with missing values. The stopping criterion
        is met once `max(abs(X_t - X_{t-1}))/max(abs(X[known_vals])) < tol`,
        where `X_t` is `X` at iteration `t`. Note that early stopping is only
        applied if `sample_posterior=False`.

    tol : float, default=1e-3
        Tolerance of the stopping condition.

    n_nearest_features : int, default=None
        Number of other features to use to estimate the missing values of
        each feature column. Nearness between features is measured using
        the absolute correlation coefficient between each feature pair (after
        initial imputation). To ensure coverage of features throughout the
        imputation process, the neighbor features are not necessarily nearest,
        but are drawn with probability proportional to correlation for each
        imputed target feature. Can provide significant speed-up when the
        number of features is huge. If `None`, all features will be used.

    initial_strategy : {'mean', 'median', 'most_frequent', 'constant'}, \
            default='mean'
        Which strategy to use to initialize the missing values. Same as the
        `strategy` parameter in :class:`~sklearn.impute.SimpleImputer`.

    fill_value : str or numerical value, default=None
        When `strategy="constant"`, `fill_value` is used to replace all
        occurrences of missing_values. For string or object data types,
        `fill_value` must be a string.
        If `None`, `fill_value` will be 0 when imputing numerical
        data and "missing_value" for strings or object data types.

        .. versionadded:: 1.3

    imputation_order : {'ascending', 'descending', 'roman', 'arabic', \
            'random'}, default='ascending'
        The order in which the features will be imputed. Possible values:

        - `'ascending'`: From features with fewest missing values to most.
        - `'descending'`: From features with most missing values to fewest.
        - `'roman'`: Left to right.
        - `'arabic'`: Right to left.
        - `'random'`: A random order for each round.

    skip_complete : bool, default=False
        If `True` then features with missing values during :meth:`transform`
        which did not have any missing values during :meth:`fit` will be
        imputed with the initial imputation method only. Set to `True` if you
        have many features with no missing values at both :meth:`fit` and
        :meth:`transform` time to save compute.

    min_value : float or array-like of shape (n_features,), default=-np.inf
        Minimum possible imputed value. Broadcast to shape `(n_features,)` if
        scalar. If array-like, expects shape `(n_features,)`, one min value for
        each feature. The default is `-np.inf`.

        .. versionchanged:: 0.23
           Added support for array-like.

    max_value : float or array-like of shape (n_features,), default=np.inf
        Maximum possible imputed value. Broadcast to shape `(n_features,)` if
        scalar. If array-like, expects shape `(n_features,)`, one max value for
        each feature. The default is `np.inf`.

        .. versionchanged:: 0.23
           Added support for array-like.

    verbose : int, default=0
        Verbosity flag, controls the debug messages that are issued
        as functions are evaluated. The higher, the more verbose. Can be 0, 1,
        or 2.

    random_state : int, RandomState instance or None, default=None
        The seed of the pseudo random number generator to use. Randomizes
        selection of estimator features if `n_nearest_features` is not `None`,
        the `imputation_order` if `random`, and the sampling from posterior if
        `sample_posterior=True`. Use an integer for determinism.
