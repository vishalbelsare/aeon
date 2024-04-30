"""Test all BaseCollection transformers comply to interface."""

import pytest

from aeon.registry import all_estimators

ALL_COLL_TRANS = all_estimators("collection-transformer", return_names=False)


@pytest.mark.parametrize("trans", ALL_COLL_TRANS)
def test_does_not_override_final_methods(trans):
    """Test does not override final methods."""
    assert "fit" not in trans.__dict__
    assert "transform" not in trans.__dict__
    assert "fit_transform" not in trans.__dict__
