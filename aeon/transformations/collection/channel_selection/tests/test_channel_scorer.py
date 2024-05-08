"""Test channel scorer channel selector."""

import pytest

from aeon.testing.mock_estimators import MockClassifierFullTags
from aeon.testing.utils.data_gen import make_example_3d_numpy
from aeon.transformations.collection.channel_selection import ChannelScorer


def test_channel_scorer():
    """Test the channel scorer."""
    # Test selects the correct number of channels
    cs = ChannelScorer(classifier=MockClassifierFullTags(), proportion=0.5)
    X, y = make_example_3d_numpy(n_channels=10)
    cs.fit(X, y)
    assert len(cs.channels_selected_) == 5
    X, y = make_example_3d_numpy(n_channels=9)
    assert len(cs.channels_selected_) == 5
    with pytest.raises(ValueError, match="proportion must be in the range 0-1"):
        ChannelScorer(classifier=MockClassifierFullTags(), proportion=1.1)
    with pytest.raises(ValueError, match="proportion must be in the range 0-1"):
        ChannelScorer(classifier=MockClassifierFullTags(), proportion=-1)
    cs = ChannelScorer(classifier=MockClassifierFullTags(), proportion=1.0)
    cs.fit(X, y)
    assert len(cs.channels_selected_) == 9
    with pytest.raises(
        ValueError, match="parameter classifier must be None or an instance of"
    ):
        ChannelScorer(classifier="FOOBAR", proportion=-1)
