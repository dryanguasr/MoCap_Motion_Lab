import json

import pytest

from seima_mocap.table_geometry import SCHEMA_VERSION, load_table_geometry


def payload():
    def section(order, points):
        return {"point_order": order,
                "points": [{"name": name, "xy": xy, "visibility": "visible"}
                           for name, xy in zip(order, points)]}
    return {"schema_version": SCHEMA_VERSION,
            "table_surface": section(["far_left", "far_right", "near_right", "near_left"],
                                     [[10, 10], [110, 10], [100, 50], [20, 50]]),
            "net": section(["top_left", "top_right", "base_right", "base_left"],
                           [[55, 5], [65, 5], [65, 45], [55, 45]])}


def test_loads_valid_geometry(tmp_path):
    path = tmp_path / "geometry.json"
    path.write_text(json.dumps(payload()))
    geometry = load_table_geometry(path)
    assert geometry["table_polygon_xy"].shape == (4, 2)
    assert geometry["net_polygon_xy"].shape == (4, 2)


def test_rejects_wrong_point_order(tmp_path):
    value = payload()
    value["table_surface"]["point_order"].reverse()
    path = tmp_path / "geometry.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="point_order"):
        load_table_geometry(path)
