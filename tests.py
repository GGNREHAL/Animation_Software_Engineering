import unittest

from utils import get_transform
from PIL import Image


TEST_IMG_PATH = "test.jpg"

class TestDataLoading(unittest.TestCase):
    def test_transform(self):
        img = Image.open(TEST_IMG_PATH)
        transform = get_transform()
        tensor  = transform(img)
        self.assertEquals(tensor.shape[0], 3)
        self.assertEquals(tensor.shape[1], 32)
        self.assertEquals(tensor.shape[2], 32)
        
        