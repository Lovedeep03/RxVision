from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load the model
model_path = '/kaggle/working/best_model.keras'
model = load_model(model_path)

# Image preprocessing
def preprocess_image(image_path, img_size=224):  # Set img_size to 224 to match model input size
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))  # Resize to 224x224
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

# Predict
image_path = '/kaggle/input/presc1/aa-Cover-hcevjn5p3605f611j8sdemdvm3-20181023004441.Medi.jpeg'
processed_img = preprocess_image(image_path)
preds = model.predict(processed_img)
pred_idx = np.argmax(preds, axis=1)[0]

# Mapping from predicted index to class name
idx_to_class = {
    0: 'Aceta',
    1: 'Ace',
    2: 'Alatrol',
    3: 'Amodis',
    4: 'Atrizin',
    5: 'Axodin',
    6: 'Azithrocin',
    7: 'Azyth',
    8: 'Az',
    9: 'Bacaid',
    10: 'Backtone',
    11: 'Baclofen',
    12: 'Baclon',
    13: 'Bacmax',
    14: 'Beklo',
    15: 'Bicozin',
    16: 'Canazole',
    17: 'Candinil',
    18: 'Cetisoft',
    19: 'Conaz',
    20: 'Dancel',
    21: 'Denixil',
    22: 'Diflu',
    23: 'Dinafex',
    24: 'Disopan',
    25: 'Esonix',
    26: 'Esoral',
    27: 'Etizin',
    28: 'Exium',
    29: 'Fenadin',
    30: 'Fexofast',
    31: 'Fexo',
    32: 'Filmet',
    33: 'Fixal',
    34: 'Flamyd',
    35: 'Flexibac',
    36: 'Flexilax',
    37: 'Flugal',
    38: 'Ketocon',
    39: 'Ketoral',
    40: 'Ketotab',
    41: 'Ketozol',
    42: 'Leptic',
    43: 'Lucan-R',
    44: 'Lumona',
    45: 'M-Kast',
    46: 'Maxima',
    47: 'Maxpro',
    48: 'Metro',
    49: 'Metsina',
    50: 'Monas',
    51: 'Montair',
    52: 'Montene',
    53: 'Montex',
    54: 'Napa Extend',
    55: 'Napa',
    56: 'Nexcap',
    57: 'Nexum',
    58: 'Nidazyl',
    59: 'Nizoder',
    60: 'Odmon',
    61: 'Omastin',
    62: 'Opton',
    63: 'Progut',
    64: 'Provair',
    65: 'Renova',
    66: 'Rhinil',
    67: 'Ritch',
    68: 'Rivotril',
    69: 'Romycin',
    70: 'Rozith',
    71: 'Sergel',
    72: 'Tamen',
    73: 'Telfast',
    74: 'Tridosil',
    75: 'Trilock',
    76: 'Vifas',
    77: 'Zithrin'
}


print("Predicted class:", idx_to_class[pred_idx])