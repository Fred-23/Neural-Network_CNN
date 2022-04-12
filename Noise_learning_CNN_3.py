import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps, ImageFilter
#https://www.researchgate.net/figure/List-of-formulas-for-the-operations-in-the-CNN-Operation-Formula_tbl1_338431622

# Charger l'image sous forme d'une matrice de pixels
img = np.array(Image.open('simba.png'))

# Générer le bruit gaussien de moyenne nulle et d'écart-type 7 (variance 49)
noise = np.random.normal(0, 7, img.shape)

# Créer l'image bruitée et l'afficher
noisy_img = Image.fromarray(img + noise).convert('L')
noisy_img.show()

# Appliquer le lissage par moyennage (fenêtre de taille 9) et afficher le résultat
noisy_img.filter(ImageFilter.BoxBlur(1)).show()

##Test des filtres pour rajouter du bruit/blur