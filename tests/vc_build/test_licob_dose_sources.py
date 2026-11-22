import numpy as np
import pytest

from vc_build.licob_dose_sources import curve_endpoints


def test_emax_can_be_identified_on_a_curve_too_short_for_ic50():
    dose = [.1, .2, .5, 1, 2]
    viability = [.9, .8, .7, .6, .4]
    estimates = curve_endpoints(dose, viability, minimum=.01, second=1, maximum=2)
    assert estimates[0] == .4
    assert .4 < estimates[1] < .9
    assert np.isnan(estimates[2])
    assert curve_endpoints(dose, viability, minimum=.01, second=1, maximum=3) is None


def test_ic50_uses_the_first_crossing_on_the_log_dose_scale():
    dose = [.01, .1, 1, 10, 20]
    curve = [.9, .7, .3, .8, .2]
    estimates = curve_endpoints(dose, curve, minimum=.01, second=10, maximum=20)
    assert estimates[0] == .2
    assert estimates[2] == pytest.approx(-.5)
    high = curve_endpoints(dose, [.8] * 5, minimum=.01, second=10, maximum=20)
    assert high[2] == pytest.approx(np.log10(20))
