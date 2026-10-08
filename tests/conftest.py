import io
import pathlib

import pytest
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
LABELS = ["daisy", "dandelion", "rose", "sunflower", "tulip"]


def jpeg(color=(200, 180, 40), size=(64, 48), **save_args) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG", **save_args)
    return buf.getvalue()


@pytest.fixture
def photo():
    return jpeg()
