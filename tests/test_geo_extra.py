# SPDX-License-Identifier: Apache-2.0
import pytest
from aidweather.geo import (
    GeoCoordinate,
    _validate_lat_lon,
    parse_any_coord_string,
)


def test_geoformat_output_precision():
    coord = GeoCoordinate(-23.55, -46.63)
    dms = coord.to_dms_str(second_precision=2)
    ddm = coord.to_ddm_str(minute_precision=4)
    dd = coord.to_dd_str(lat_precision=1, lon_precision=1)
    # Verify format patterns
    assert "°" in dms[0] and "'" in dms[0] and '"' in dms[0]
    assert "°" in ddm[0] and "'" in ddm[0]
    assert "°" in dd[0]


def test_geo_parse_any_coord_invalid_hemisphere():
    # hemisphere overrides sign
    val = parse_any_coord_string("-23.55 S", is_lat=True)
    assert val == pytest.approx(-23.55)
    val2 = parse_any_coord_string("23.55 N", is_lat=True)
    assert val2 == pytest.approx(23.55)


def test_validate_lat_lon_bounds():
    _validate_lat_lon(90, 180)
    _validate_lat_lon(-90, -180)
    with pytest.raises(ValueError):
        _validate_lat_lon(90.01, 0)
    with pytest.raises(ValueError):
        _validate_lat_lon(0, 180.01)
