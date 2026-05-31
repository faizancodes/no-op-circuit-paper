        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            The anomaly score of the input samples.
            The lower, the more abnormal. Negative scores represent outliers,
            positive scores represent inliers.
        """
        # We subtract self.offset_ to make 0 be the threshold value for being
        # an outlier:

        return self.score_samples(X) - self.offset_

    def score_samples(self, X):
        """
        Opposite of the anomaly score defined in the original paper.

        The anomaly score of an input sample is computed as
        the mean anomaly score of the trees in the forest.

        The measure of normality of an observation given a tree is the depth
        of the leaf containing this observation, which is equivalent to
        the number of splittings required to isolate this point. In case of
        several observations n_left in the leaf, the average path length of
        a n_left samples isolation tree is added.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            The anomaly score of the input samples.
            The lower, the more abnormal.
        """
        # Check data
        X = self._validate_data(X, accept_sparse="csr", dtype=np.float32, reset=False)

        return self._score_samples(X)

    def _score_samples(self, X):
        """Private version of score_samples without input validation.

        Input validation would remove feature names, so we disable it.
        """
        # Code structure from ForestClassifier/predict_proba

        check_is_fitted(self)

        # Take the opposite of the scores as bigger is better (here less abnormal)
        return -self._compute_chunked_score_samples(X)

    def _compute_chunked_score_samples(self, X):
        n_samples = _num_samples(X)

        if self._max_features == X.shape[1]:
            subsample_features = False
        else:
            subsample_features = True

        # We get as many rows as possible within our working_memory budget
        # (defined by sklearn.get_config()['working_memory']) to store
        # self._max_features in each row during computation.
        #
        # Note:
        #  - this will get at least 1 row, even if 1 row of score will
        #    exceed working_memory.
        #  - this does only account for temporary memory usage while loading
        #    the data needed to compute the scores -- the returned scores
        #    themselves are 1D.

        chunk_n_rows = get_chunk_n_rows(
            row_bytes=16 * self._max_features, max_n_rows=n_samples
        )
        slices = gen_batches(n_samples, chunk_n_rows)

        scores = np.zeros(n_samples, order="f")

        for sl in slices:
            # compute score on the slices of test samples:
            scores[sl] = self._compute_score_samples(X[sl], subsample_features)

        return scores

    def _compute_score_samples(self, X, subsample_features):
        """
        Compute the score of each samples in X going through the extra trees.

        Parameters
        ----------
        X : array-like or sparse matrix
            Data matrix.
