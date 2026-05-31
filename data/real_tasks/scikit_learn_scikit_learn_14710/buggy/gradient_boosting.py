            print("{:<32} {:.3f}s".format('Time spent predicting:',
                                          acc_prediction_time))

        self.train_score_ = np.asarray(self.train_score_)
        self.validation_score_ = np.asarray(self.validation_score_)
        del self._in_fit  # hard delete so we're sure it can't be used anymore
        return self

    def _is_fitted(self):
        return len(getattr(self, '_predictors', [])) > 0

    def _clear_state(self):
        """Clear the state of the gradient boosting model."""
        for var in ('train_score_', 'validation_score_', '_rng'):
            if hasattr(self, var):
                delattr(self, var)

    def _get_small_trainset(self, X_binned_train, y_train, seed):
        """Compute the indices of the subsample set and return this set.

        For efficiency, we need to subsample the training set to compute scores
        with scorers.
        """
        subsample_size = 10000
        if X_binned_train.shape[0] > subsample_size:
            indices = np.arange(X_binned_train.shape[0])
            stratify = y_train if is_classifier(self) else None
            indices = resample(indices, n_samples=subsample_size,
                               replace=False, random_state=seed,
                               stratify=stratify)
            X_binned_small_train = X_binned_train[indices]
            y_small_train = y_train[indices]
            X_binned_small_train = np.ascontiguousarray(X_binned_small_train)
            return X_binned_small_train, y_small_train
        else:
            return X_binned_train, y_train

    def _check_early_stopping_scorer(self, X_binned_small_train, y_small_train,
                                     X_binned_val, y_val):
        """Check if fitting should be early-stopped based on scorer.

        Scores are computed on validation data or on training data.
        """
        self.train_score_.append(
            self.scorer_(self, X_binned_small_train, y_small_train)
        )

        if self._use_validation_data:
            self.validation_score_.append(
                self.scorer_(self, X_binned_val, y_val)
            )
            return self._should_stop(self.validation_score_)
        else:
            return self._should_stop(self.train_score_)

    def _check_early_stopping_loss(self,
                                   raw_predictions,
                                   y_train,
                                   raw_predictions_val,
                                   y_val):
        """Check if fitting should be early-stopped based on loss.

        Scores are computed on validation data or on training data.
        """

        self.train_score_.append(
            -self.loss_(y_train, raw_predictions)
        )

        if self._use_validation_data:
            self.validation_score_.append(
                -self.loss_(y_val, raw_predictions_val)
            )
            return self._should_stop(self.validation_score_)
        else:
            return self._should_stop(self.train_score_)

    def _should_stop(self, scores):
        """
        Return True (do early stopping) if the last n scores aren't better
        than the (n-1)th-to-last score, up to some tolerance.
        """
        reference_position = self.n_iter_no_change + 1
        if len(scores) < reference_position:
            return False

        # A higher score is always better. Higher tol means that it will be
        # harder for subsequent iteration to be considered an improvement upon
        # the reference score, and therefore it is more likely to early stop
        # because of the lack of significant improvement.
        tol = 0 if self.tol is None else self.tol
