"""Pelt (Linearly penalized segmentation) Segmenter."""

__maintainer__ = []
__all__ = ["PeltSegmenter"]

import numpy as np
import pandas as pd

from aeon.segmentation.base import BaseSegmenter
from aeon.utils.validation._dependencies import _check_soft_dependencies


class PeltSegmenter(BaseSegmenter):
    """Pelt (Linearly penalized segmentation) Segmenter.

    From the Ruptures documentation:
    "Because the enumeration of all possible partitions impossible, the algorithm relies
    on a pruning rule. Many indexes are discarded, greatly reducing the computational
    cost while retaining the ability to find the optimal segmentation. The
    implementation follows [Killick2012]. In addition, under certain conditions on the
    change point repartition, the avarage computational complexity is of the order of
    O(CKn), where K is the number of change points to detect, n the number of samples
    and C the complexity of calling the considered cost function on one sub-signal.

    Consequently, piecewise constant models (model=l2) are significantly faster than
    linear or autoregressive models."

    Parameters
    ----------
    n_cps : int, default = 1
        The number of change points to search.
    model : str, default = "l2"
        Segment model to use. Options are "l1", "l2", "rbf", etc.
        (see ruptures documentation for available models).
    min_size : int, default = 2,
        Minimum segment length. Defaults to 2 samples.
        (see ruptures documentation for additional information).
    jump : int, default = 5,
        Subsample (one every jump points). Defaults to 5.
        (see ruptures documentation for additional information).
    pen : int, default = 2
        penalty value.
        (see ruptures documentation for additional information).

    References
    ----------
    .. [Killick2012] Killick, R., Fearnhead, P., & Eckley, I. (2012). Optimal detection
    of changepoints with a linear computational cost. Journal of the American
    Statistical Association, 107(500), 1590–1598.

    Examples
    --------
    >>> from aeon.segmentation import PeltSegmenter
    >>> from aeon.datasets import load_gun_point_segmentation
    >>> X, true_period_size, cps = load_gun_point_segmentation()
    >>> pelt = PeltSegmenter(10, n_cps=1)  # doctest: +SKIP
    >>> found_cps = pelt.fit_predict(X)  # doctest: +SKIP
    """

    _tags = {
        "fit_is_empty": True,
        "python_dependencies": "ruptures",
    }

    def __init__(self, n_cps=1, model="l2", min_size=2, jump=5, pen=2):
        self.n_cps = n_cps
        self.min_size = min_size
        self.jump = jump
        self.model = model
        self.pen = pen
        super().__init__(n_segments=n_cps + 1, axis=1)

    def _predict(self, X: np.ndarray):
        """Create annotations on test/deployment data.

        Parameters
        ----------
        X : np.ndarray
            1D time series to be segmented.

        Returns
        -------
        list
            List of change points found in X.
        """
        X = X.squeeze()
        self.found_cps = self._run_pelt(X)
        return self.found_cps

    def get_fitted_params(self):
        """Get fitted parameters.

        Returns
        -------
        fitted_params : dict
        """
        return {}

    def _run_pelt(self, X):
        _check_soft_dependencies("ruptures", severity="error")
        import ruptures as rpt

        pelt = rpt.Pelt(model=self.model, min_size=self.min_size, jump=self.jump).fit(X)
        self.found_cps = np.array(
            pelt.predict(pen=self.pen)[: self.n_cps], dtype=np.int64
        )

        return self.found_cps

    def _get_interval_series(self, X, found_cps):
        """Get the segmentation results based on the found change points.

        Parameters
        ----------
        X :         array-like, shape = [n]
           Univariate time-series data to be segmented.
        found_cps : array-like, shape = [n_cps] The found change points found

        Returns
        -------
        IntervalIndex:
            Segmentation based on found change points

        """
        cps = np.array(found_cps)
        start = np.insert(cps, 0, 0)
        end = np.append(cps, len(X))
        return pd.IntervalIndex.from_arrays(start, end)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        return {"n_cps": 1}
