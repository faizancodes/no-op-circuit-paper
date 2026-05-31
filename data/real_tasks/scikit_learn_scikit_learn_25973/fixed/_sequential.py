"""
Sequential feature selection
"""
from numbers import Integral, Real

import numpy as np

import warnings

from ._base import SelectorMixin
from ..base import BaseEstimator, MetaEstimatorMixin, clone, is_classifier
from ..utils._param_validation import HasMethods, Hidden, Interval, StrOptions
from ..utils._param_validation import RealNotInt
from ..utils._tags import _safe_tags
from ..utils.validation import check_is_fitted
from ..model_selection import cross_val_score, check_cv
from ..metrics import get_scorer_names


class SequentialFeatureSelector(SelectorMixin, MetaEstimatorMixin, BaseEstimator):
    """Transformer that performs Sequential Feature Selection.

    This Sequential Feature Selector adds (forward selection) or
    removes (backward selection) features to form a feature subset in a
    greedy fashion. At each stage, this estimator chooses the best feature to
    add or remove based on the cross-validation score of an estimator. In
    the case of unsupervised learning, this Sequential Feature Selector
    looks only at the features (X), not the desired outputs (y).

    Read more in the :ref:`User Guide <sequential_feature_selection>`.

    .. versionadded:: 0.24

    Parameters
    ----------
    estimator : estimator instance
        An unfitted estimator.

    n_features_to_select : "auto", int or float, default='warn'
        If `"auto"`, the behaviour depends on the `tol` parameter:

        - if `tol` is not `None`, then features are selected until the score
          improvement does not exceed `tol`.
        - otherwise, half of the features are selected.

        If integer, the parameter is the absolute number of features to select.
        If float between 0 and 1, it is the fraction of features to select.

        .. versionadded:: 1.1
           The option `"auto"` was added in version 1.1.

        .. deprecated:: 1.1
           The default changed from `None` to `"warn"` in 1.1 and will become
           `"auto"` in 1.3. `None` and `'warn'` will be removed in 1.3.
           To keep the same behaviour as `None`, set
           `n_features_to_select="auto" and `tol=None`.

    tol : float, default=None
        If the score is not incremented by at least `tol` between two
