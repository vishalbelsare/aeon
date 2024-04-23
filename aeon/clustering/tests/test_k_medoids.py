"""Tests for time series k-medoids."""

import numpy as np
from sklearn import metrics
from sklearn.utils import check_random_state
import pytest

from aeon.clustering._k_medoids import TimeSeriesKMedoids
from aeon.distances import euclidean_distance, euclidean_pairwise_distance
from aeon.testing.utils.data_gen import make_example_3d_numpy


def test_kmedoids_uni():
    """Test implementation of Kmedoids."""
    X_train, y_train = make_example_3d_numpy(10, 1, 10, random_state=1)
    X_test, y_test = make_example_3d_numpy(10, 1, 10, random_state=2)

    num_points = 10

    X_train = X_train[:num_points]
    y_train = y_train[:num_points]
    X_test = X_test[:num_points]
    y_test = y_test[:num_points]
    _alternate_uni_medoids(X_train, y_train, X_test, y_test)
    _pam_uni_medoids(X_train, y_train, X_test, y_test)
    # precomputed
    _alternate_uni_medoids(X_train, y_train, X_test, y_test, use_precomputed=True)
    _pam_uni_medoids(X_train, y_train, X_test, y_test, use_precomputed=True)


def test_kmedoids_multi():
    """Test implementation of Kmedoids for multivariate."""
    X_train, y_train = make_example_3d_numpy(10, 10, 10, random_state=1)
    X_test, y_test = make_example_3d_numpy(10, 10, 10, random_state=2)

    num_points = 10

    X_train = X_train[:num_points]
    y_train = y_train[:num_points]
    X_test = X_test[:num_points]
    y_test = y_test[:num_points]
    _alternate_multi_medoids(X_train, y_train, X_test, y_test)
    _pam_multi_medoids(X_train, y_train, X_test, y_test)
    # precomputed
    _alternate_multi_medoids(X_train, y_train, X_test, y_test, use_precomputed=True)
    _pam_multi_medoids(X_train, y_train, X_test, y_test, use_precomputed=True)


def _pam_uni_medoids(X_train, y_train, X_test, y_test, use_precomputed=False):
    if use_precomputed:
        X_train = euclidean_pairwise_distance(X_train)
        kmedoids = TimeSeriesKMedoids(
            random_state=1,
            n_init=2,
            max_iter=5,
            init_algorithm="first",
            distance="precomputed",
            method="pam",
        )
        train_medoids_result = kmedoids.fit_predict(X_train)
        train_score = metrics.rand_score(y_train, train_medoids_result)
        with pytest.raises(ValueError):
            kmedoids.predict(X_test)
        assert np.array_equal(train_medoids_result, [5, 1, 2, 3, 4, 5, 6, 7, 0, 3])
        assert train_score == 0.6
        assert np.isclose(kmedoids.inertia_, 3.04125095258111)
        assert kmedoids.n_iter_ == 2
        assert np.array_equal(kmedoids.labels_, [5, 1, 2, 3, 4, 5, 6, 7, 0, 3])
        assert kmedoids.cluster_centers_ is None
        assert isinstance(kmedoids.center_indexes_, np.ndarray)
    else:
        kmedoids = TimeSeriesKMedoids(
            random_state=1,
            n_init=2,
            max_iter=5,
            init_algorithm="first",
            distance="euclidean",
            method="pam",
        )
        train_medoids_result = kmedoids.fit_predict(X_train)
        train_score = metrics.rand_score(y_train, train_medoids_result)
        test_medoids_result = kmedoids.predict(X_test)
        test_score = metrics.rand_score(y_test, test_medoids_result)
        proba = kmedoids.predict_proba(X_test)
        assert np.array_equal(test_medoids_result, [3, 4, 6, 3, 0, 7, 0, 6, 6, 3])
        assert np.array_equal(train_medoids_result, [5, 1, 2, 3, 4, 5, 6, 7, 0, 3])
        assert test_score == 0.6222222222222222
        assert train_score == 0.6
        assert np.isclose(kmedoids.inertia_, 3.04125095258111)
        assert kmedoids.n_iter_ == 2
        assert np.array_equal(kmedoids.labels_, [5, 1, 2, 3, 4, 5, 6, 7, 0, 3])
        assert isinstance(kmedoids.cluster_centers_, np.ndarray)
        for val in proba:
            assert np.count_nonzero(val == 1.0) == 1


def _alternate_uni_medoids(X_train, y_train, X_test, y_test, use_precomputed=False):
    if use_precomputed:
        X_train = euclidean_pairwise_distance(X_train)
        kmedoids = TimeSeriesKMedoids(
            random_state=1,
            n_init=2,
            max_iter=5,
            method="alternate",
            init_algorithm="first",
            distance="precomputed",
        )
        train_medoids_result = kmedoids.fit_predict(X_train)
        train_score = metrics.rand_score(y_train, train_medoids_result)
        with pytest.raises(ValueError):
            kmedoids.predict(X_test)
        assert np.array_equal(train_medoids_result, [0, 1, 2, 3, 4, 5, 6, 7, 1, 3])
        assert train_score == 0.6
        assert np.isclose(kmedoids.inertia_, 6.571247130721869)
        assert kmedoids.n_iter_ == 2
        assert np.array_equal(kmedoids.labels_, [0, 1, 2, 3, 4, 5, 6, 7, 1, 3])
        assert kmedoids.cluster_centers_ is None
        assert isinstance(kmedoids.center_indexes_, np.ndarray)
    else:
        kmedoids = TimeSeriesKMedoids(
            random_state=1,
            n_init=2,
            max_iter=5,
            method="alternate",
            init_algorithm="first",
            distance="euclidean",
        )
        train_medoids_result = kmedoids.fit_predict(X_train)
        train_score = metrics.rand_score(y_train, train_medoids_result)
        test_medoids_result = kmedoids.predict(X_test)
        test_score = metrics.rand_score(y_test, test_medoids_result)
        proba = kmedoids.predict_proba(X_test)
        assert np.array_equal(test_medoids_result, [3, 4, 6, 3, 7, 7, 6, 6, 6, 3])
        assert np.array_equal(train_medoids_result, [0, 1, 2, 3, 4, 5, 6, 7, 1, 3])
        assert test_score == 0.6888888888888889
        assert train_score == 0.6
        assert np.isclose(kmedoids.inertia_, 6.571247130721869)
        assert kmedoids.n_iter_ == 2
        assert np.array_equal(kmedoids.labels_, [0, 1, 2, 3, 4, 5, 6, 7, 1, 3])
        assert isinstance(kmedoids.cluster_centers_, np.ndarray)
        for val in proba:
            assert np.count_nonzero(val == 1.0) == 1


def _pam_multi_medoids(X_train, y_train, X_test, y_test, use_precomputed=False):
    if use_precomputed:
        X_train = euclidean_pairwise_distance(X_train)

        kmedoids = TimeSeriesKMedoids(
            random_state=1,
            n_init=2,
            max_iter=5,
            init_algorithm="first",
            distance="precomputed",
            method="pam",
        )
        train_medoids_result = kmedoids.fit_predict(X_train)
        train_score = metrics.rand_score(y_train, train_medoids_result)
        with pytest.raises(ValueError):
            kmedoids.predict(X_test)
        assert np.array_equal(train_medoids_result, [7, 1, 2, 3, 4, 7, 6, 7, 5, 0])
        assert train_score == 0.6
        assert np.isclose(kmedoids.inertia_, 15.232292770407273)
        assert kmedoids.n_iter_ == 3
        assert np.array_equal(kmedoids.labels_, [7, 1, 2, 3, 4, 7, 6, 7, 5, 0])
        assert kmedoids.cluster_centers_ is None
        assert isinstance(kmedoids.center_indexes_, np.ndarray)
    else:
        kmedoids = TimeSeriesKMedoids(
            random_state=1,
            n_init=2,
            max_iter=5,
            init_algorithm="first",
            distance="euclidean",
            method="pam",
        )
        train_medoids_result = kmedoids.fit_predict(X_train)
        train_score = metrics.rand_score(y_train, train_medoids_result)
        test_medoids_result = kmedoids.predict(X_test)
        test_score = metrics.rand_score(y_test, test_medoids_result)
        proba = kmedoids.predict_proba(X_test)
        assert np.array_equal(test_medoids_result, [7, 7, 3, 7, 7, 7, 0, 7, 5, 7])
        assert np.array_equal(train_medoids_result, [7, 1, 2, 3, 4, 7, 6, 7, 5, 0])
        assert test_score == 0.6666666666666666
        assert train_score == 0.6
        assert np.isclose(kmedoids.inertia_, 15.232292770407273)
        assert kmedoids.n_iter_ == 3
        assert np.array_equal(kmedoids.labels_, [7, 1, 2, 3, 4, 7, 6, 7, 5, 0])
        assert isinstance(kmedoids.cluster_centers_, np.ndarray)
        for val in proba:
            assert np.count_nonzero(val == 1.0) == 1


def _alternate_multi_medoids(X_train, y_train, X_test, y_test, use_precomputed=False):
    if use_precomputed:
        X_train = euclidean_pairwise_distance(X_train)

        kmedoids = TimeSeriesKMedoids(
            random_state=1,
            n_init=2,
            max_iter=5,
            init_algorithm="first",
            method="alternate",
            distance="precomputed",
        )
        train_medoids_result = kmedoids.fit_predict(X_train)
        train_score = metrics.rand_score(y_train, train_medoids_result)
        with pytest.raises(ValueError):
            kmedoids.predict(X_test)
        assert np.array_equal(train_medoids_result, [0, 1, 2, 3, 4, 5, 6, 7, 4, 7])
        assert train_score == 0.5777777777777777
        assert np.isclose(kmedoids.inertia_, 23.492613611209528)
        assert kmedoids.n_iter_ == 2
        assert np.array_equal(kmedoids.labels_, [0, 1, 2, 3, 4, 5, 6, 7, 4, 7])
        assert kmedoids.cluster_centers_ is None
        assert isinstance(kmedoids.center_indexes_, np.ndarray)
    else:
        kmedoids = TimeSeriesKMedoids(
            random_state=1,
            n_init=2,
            max_iter=5,
            init_algorithm="first",
            method="alternate",
            distance="euclidean",
        )
        train_medoids_result = kmedoids.fit_predict(X_train)
        train_score = metrics.rand_score(y_train, train_medoids_result)
        test_medoids_result = kmedoids.predict(X_test)
        test_score = metrics.rand_score(y_test, test_medoids_result)
        proba = kmedoids.predict_proba(X_test)
        assert np.array_equal(test_medoids_result, [0, 7, 3, 5, 7, 0, 5, 5, 7, 7])
        assert np.array_equal(train_medoids_result, [0, 1, 2, 3, 4, 5, 6, 7, 4, 7])
        assert test_score == 0.5111111111111111
        assert train_score == 0.5777777777777777
        assert np.isclose(kmedoids.inertia_, 23.492613611209528)
        assert kmedoids.n_iter_ == 2
        assert np.array_equal(kmedoids.labels_, [0, 1, 2, 3, 4, 5, 6, 7, 4, 7])
        assert isinstance(kmedoids.cluster_centers_, np.ndarray)
        for val in proba:
            assert np.count_nonzero(val == 1.0) == 1


def check_value_in_every_cluster(num_clusters, initial_medoids):
    """Check that every cluster has at least one value."""
    original_length = len(initial_medoids)
    assert original_length == num_clusters
    if isinstance(initial_medoids, np.ndarray):
        for i in range(len(initial_medoids)):
            curr = initial_medoids[i]
            for j in range(len(initial_medoids)):
                if i == j:
                    continue
                other = initial_medoids[j]
                assert not np.array_equal(curr, other)
    else:
        assert original_length == len(set(initial_medoids))


def test_medoids_init():
    """Test implementation of Kmedoids."""
    X_train = make_example_3d_numpy(10, 1, 10, return_y=False, random_state=1)
    X_train = X_train[:10]

    num_clusters = 8
    kmedoids = TimeSeriesKMedoids(
        random_state=1,
        n_init=1,
        max_iter=5,
        init_algorithm="first",
        distance="euclidean",
        n_clusters=num_clusters,
    )
    kmedoids._random_state = check_random_state(kmedoids.random_state)
    kmedoids._distance_cache = np.full((len(X_train), len(X_train)), np.inf)
    kmedoids._distance_callable = euclidean_distance
    first_medoids_result = kmedoids._first_center_initializer(X_train)
    check_value_in_every_cluster(num_clusters, first_medoids_result)
    random_medoids_result = kmedoids._random_center_initializer(X_train)
    check_value_in_every_cluster(num_clusters, random_medoids_result)
    kmedoids_plus_plus_medoids_result = kmedoids._kmedoids_plus_plus_center_initializer(
        X_train
    )
    check_value_in_every_cluster(num_clusters, kmedoids_plus_plus_medoids_result)
    kmedoids_build_result = kmedoids._pam_build_center_initializer(X_train)
    check_value_in_every_cluster(num_clusters, kmedoids_build_result)

    # Test setting manual init centres
    num_clusters = 8
    custom_init_centres = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    kmedoids = TimeSeriesKMedoids(
        random_state=1,
        n_init=1,
        max_iter=5,
        init_algorithm=custom_init_centres,
        distance="euclidean",
        n_clusters=num_clusters,
    )
    kmedoids.fit(X_train)
    assert np.array_equal(kmedoids.cluster_centers_, X_train[custom_init_centres])


def _get_model_centres(data, distance, method="pam", distance_params=None):
    """Get the centres of a model."""
    model = TimeSeriesKMedoids(
        random_state=1,
        method=method,
        n_init=2,
        n_clusters=2,
        init_algorithm="random",
        distance=distance,
        distance_params=distance_params,
    )
    model.fit(data)
    return model.cluster_centers_


def test_custom_distance_params():
    """Test kmedoids custom distance parameters."""
    X_train, y_train = make_example_3d_numpy(10, 1, 10, random_state=1)

    num_test_values = 10
    data = X_train[0:num_test_values]

    # Test passing distance param
    default_dist = _get_model_centres(data, distance="msm")
    custom_params_dist = _get_model_centres(
        data, distance="msm", distance_params={"window": 0.2}
    )
    assert not np.array_equal(default_dist, custom_params_dist)
