import unittest
from utils import get_transform
from PIL import Image

import numpy as np
from MNIST_Fashion_Denoiser import Autoencoder()

class TestAutoencoder(unittest.TestCase):
    def setUp(self):
        self.model = Autoencoder()
        
    def trainloader(self):
        test_image = np.random.rand(28, 28)
        output = self.model.predict(trainloader)
        self.assertTrue(np.shape(output) == (28, 28))
        
    def testloader(self):
        test_image = np.random.rand(28, 28)
        test_label = np.random.randint(0, 10)
        score = self.model.score(testloader)
        self.assertTrue(score >= 0 and score <= 1)
        
    def val_loader(self):
        test_image = np.random.rand(28, 28)
        denoised_image = self.model.denoise(testloader)
        self.assertTrue(np.shape(denoised_image) == (28, 28))
        
if __name__ == '__init__':
    unittest.main()



        