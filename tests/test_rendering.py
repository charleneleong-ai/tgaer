from __future__ import annotations

from tgaer.envs.arc_agi3.rendering import ARC_PALETTE, grid_to_png_data_url, grid_to_rgb

FRAME = [[[0, 1], [2, 15]]]


class TestGridToRgb:
    def test_maps_indices_through_palette(self):
        rgb = grid_to_rgb(FRAME)
        assert rgb.shape == (2, 2, 3)
        assert tuple(rgb[0, 0]) == ARC_PALETTE[0]
        assert tuple(rgb[1, 1]) == ARC_PALETTE[15]

    def test_empty_is_none(self):
        assert grid_to_rgb(None) is None
        assert grid_to_rgb([]) is None


class TestPngDataUrl:
    def test_renders_png_data_url(self):
        url = grid_to_png_data_url(FRAME)
        assert url.startswith("data:image/png;base64,") and len(url) > 40

    def test_empty_is_none(self):
        assert grid_to_png_data_url(None) is None
