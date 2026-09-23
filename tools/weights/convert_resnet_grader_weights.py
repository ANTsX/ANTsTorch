#!/usr/bin/env python3
"""
convert_resnet_grader_weights.py
Script d'extraction structurelle stricte des poids ResNet (Keras -> PyTorch).
Gère l'inversion topologique 'shortcut'/'conv3' et la suppression des biais aléatoires.
"""

import argparse
import os
import sys
import numpy as np
import torch

try:
    import antspynet
    import antstorch
except ImportError:
    print("Veuillez installer antspynet et antstorch dans votre environnement.")
    sys.exit(1)

def assign_conv(pt_conv, k_w, k_b_list):
    # Keras (kD, kH, kW, inC, outC) -> PyTorch (outC, inC, kD, kH, kW)
    torch_w = np.transpose(k_w, (4, 3, 0, 1, 2))
    pt_conv.weight.data.copy_(torch.from_numpy(torch_w).float())
    
    if pt_conv.bias is not None:
        if len(k_b_list) > 0:
            pt_conv.bias.data.copy_(torch.from_numpy(k_b_list.pop(0)).float())
        else:
            pt_conv.bias.data.zero_() # Supprime le bruit aléatoire de PyTorch

def assign_bn(pt_bn, gamma, beta, mean, var):
    pt_bn.weight.data.copy_(torch.from_numpy(gamma).float())
    pt_bn.bias.data.copy_(torch.from_numpy(beta).float())
    pt_bn.running_mean.data.copy_(torch.from_numpy(mean).float())
    pt_bn.running_var.data.copy_(torch.from_numpy(var).float())

def convert_resnet_grader(out_path: str, verbose: bool = True):
    if verbose: print("1. Initialisation de Keras et téléchargement des poids...")
    
    kmodel = antspynet.create_resnet_model_3d(
        [None, None, None, 1], lowest_resolution=32, number_of_outputs=4,
        cardinality=1, squeeze_and_excite=False
    )
    kmodel.load_weights(antspynet.get_pretrained_network("resnet_grader"))

    k_conv_w, k_conv_b = [], []
    k_bn_g, k_bn_b, k_bn_m, k_bn_v = [], [], [], []
    k_dense_w, k_dense_b = [], []

    for layer in kmodel.layers:
        ltype = layer.__class__.__name__.lower()
        if 'conv3d' in ltype:
            w = layer.get_weights()
            if len(w) >= 1: k_conv_w.append(w[0])
            if len(w) == 2: k_conv_b.append(w[1])
        elif 'batchnormalization' in ltype:
            w = layer.get_weights()
            k_bn_g.append(w[0]); k_bn_b.append(w[1])
            k_bn_m.append(w[2]); k_bn_v.append(w[3])
        elif 'dense' in ltype:
            w = layer.get_weights()
            k_dense_w.append(w[0])
            if len(w) == 2: k_dense_b.append(w[1])

    if verbose: print("2. Mapping structurel strict vers PyTorch...")

    tmodel = antstorch.create_resnet_model_3d(
        input_channel_size=1, lowest_resolution=32, number_of_outputs=4,
        cardinality=1, squeeze_and_excite=False
    )
    tmodel.eval()

    # --- 1. Init Conv ---
    assign_conv(tmodel.init_conv[0], k_conv_w.pop(0), k_conv_b)
    assign_bn(tmodel.init_conv[1][0], k_bn_g.pop(0), k_bn_b.pop(0), k_bn_m.pop(0), k_bn_v.pop(0))

    # --- 2. Blocs Résiduels ---
    for block in tmodel.model_residual_layers:
        assign_conv(block.conv1[0], k_conv_w.pop(0), k_conv_b)
        assign_bn(block.conv1[1][0], k_bn_g.pop(0), k_bn_b.pop(0), k_bn_m.pop(0), k_bn_v.pop(0))
        
        assign_conv(block.conv2[0], k_conv_w.pop(0), k_conv_b)
        assign_bn(block.conv2[1][0], k_bn_g.pop(0), k_bn_b.pop(0), k_bn_m.pop(0), k_bn_v.pop(0))

        if block.shortcut is not None:
            # Ordre du graphe Keras : Shortcut AVANT Conv3
            sc_w = k_conv_w.pop(0)
            sc_g, sc_b, sc_m, sc_v = k_bn_g.pop(0), k_bn_b.pop(0), k_bn_m.pop(0), k_bn_v.pop(0)
            
            c3_w = k_conv_w.pop(0)
            c3_g, c3_b, c3_m, c3_v = k_bn_g.pop(0), k_bn_b.pop(0), k_bn_m.pop(0), k_bn_v.pop(0)

            # Assignation PyTorch
            assign_conv(block.shortcut[0], sc_w, k_conv_b)
            assign_bn(block.shortcut[1], sc_g, sc_b, sc_m, sc_v)
            
            assign_conv(block.conv3[0], c3_w, k_conv_b)
            assign_bn(block.conv3[1], c3_g, c3_b, c3_m, c3_v)
        else:
            assign_conv(block.conv3[0], k_conv_w.pop(0), k_conv_b)
            assign_bn(block.conv3[1], k_bn_g.pop(0), k_bn_b.pop(0), k_bn_m.pop(0), k_bn_v.pop(0))

    # --- 3. Dense (Softmax) ---
    keras_dense = k_dense_w.pop(0)
    tmodel.dense[0].weight.data.copy_(torch.from_numpy(np.transpose(keras_dense, (1, 0))).float())
    if len(k_dense_b) > 0:
        tmodel.dense[0].bias.data.copy_(torch.from_numpy(k_dense_b.pop(0)).float())
    elif tmodel.dense[0].bias is not None:
        tmodel.dense[0].bias.data.zero_()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save(tmodel.state_dict(), out_path)
    if verbose: print(f"3. Conversion terminée et sécurisée : {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=os.path.expanduser("~/.antstorch/resnet_grader_pytorch.pt"))
    parser.add_argument("--quiet", action="store_false", dest="verbose")
    args = parser.parse_args()
    convert_resnet_grader(out_path=args.out, verbose=args.verbose)