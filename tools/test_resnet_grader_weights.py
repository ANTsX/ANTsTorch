import ants
import torch
import numpy as np
import antspynet
import antstorch

# 1. Chargement et prétraitement basique (sans aléatoire)
image = ants.image_read(antspynet.get_antsxnet_data("mprage_hippmapp3r"))
t1 = ants.iMath(image - image.min(), "Normalize")

# 2. Instanciation des modèles
weights_k = antspynet.get_pretrained_network("resnet_grader")
mdl_k = antspynet.create_resnet_model_3d([None, None, None, 1], lowest_resolution=32, number_of_outputs=4, cardinality=1, squeeze_and_excite=False)
mdl_k.load_weights(weights_k)

weights_pt = antstorch.get_pretrained_network("resnet_grader_pytorch") # ou votre chemin local
mdl_pt = antstorch.create_resnet_model_3d(input_channel_size=1, lowest_resolution=32, number_of_outputs=4, cardinality=1, squeeze_and_excite=False)
mdl_pt.load_state_dict(torch.load(weights_pt, map_location="cpu"))
mdl_pt.eval()

# 3. Formatage du tenseur unique
xarr = t1.numpy()
xarr = ants.resample_image(t1, (2,2,2)).numpy() # Réduction rapide pour le test
xarr_k = np.reshape(xarr, [1] + list(xarr.shape) + [1])
xarr_pt = torch.from_numpy(np.reshape(xarr, [1, 1] + list(xarr.shape))).float()

# 4. Comparaison des vecteurs Softmax
preds_k = mdl_k.predict(xarr_k, verbose=0)

with torch.no_grad():
    preds_pt = mdl_pt(xarr_pt).cpu().numpy()

print("Vecteur Probabilités Keras   :", preds_k[0])
print("Vecteur Probabilités PyTorch :", preds_pt[0])