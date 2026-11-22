from types import SimpleNamespace

import numpy as np
import pytest

from vc_build.wong_tissue import paired_tpm


def test_tissue_uses_the_matching_parental_column_for_each_organoid():
    ids = [f"PDTO_{n}" for n in range(10)]
    parents = [f"PDT_{n}_ParentalTissue" for n in reversed(range(10))]
    header = ["Gene symbol", *ids, *parents]
    row = ["A", *range(100, 110), *reversed(range(10))]
    data = paired_tpm(SimpleNamespace(values=[header, row]), ["A"], ids)
    assert list(data.index) == ids
    np.testing.assert_allclose(data["A"], np.log2(1 + np.arange(10)))
    broken = ["Gene symbol", *ids, *parents[:-1], "PDT_0"]
    with pytest.raises(ValueError, match="matching parental-tumor"):
        paired_tpm(SimpleNamespace(values=[broken, row]), ["A"], ids)
