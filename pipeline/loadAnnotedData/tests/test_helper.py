"""
Unit tests for pipeline.loadAnnotedData.tests.test_helper

To run doctest:
    pytest --doctest-modules vitm/models/patch_embed.py
"""

from ..helper import read_coco_file

# Incomplete do not run
def test_read_coco_file():
    """
        The test requires the visual confirmation
    """
    read_coco_file(0, 2)