"""pytest entry points of the follow-up integrity fixtures (plan Appendix C 3, 5-8).
The fixtures that need the server data (C5, C6, real-graph part of C7) are skipped when the data environment is absent;
on the server they run in the r7f_fixtures job (python -m src.r7_followup_v1.fixtures)."""
import os
import pytest

from src.r7_followup_v1 import fixtures as fx


def test_c3_retrieval_ties():
    fx.fx_retrieval()


def test_c7_paths_synthetic():
    pytest.importorskip("torch")
    fx.fx_paths(real_graph="R7_KG" in os.environ)


def test_c8_target_difference_zero():
    pytest.importorskip("torch")
    fx.fx_target_difference()


@pytest.mark.skipif("R4_PROC" not in os.environ, reason="needs the server unit store")
def test_c5_c6_fold_null_contract():
    fx.fx_fold_and_null()
