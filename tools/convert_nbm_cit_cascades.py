#!/usr/bin/env python3
"""
convert_nbm_cit_cascades.py

Script dédié pour convertir les poids des modèles en cascade (NBM et CIT168)
depuis ANTsPyNet (Keras) vers ANTsTorch (PyTorch).
"""

import os
import torch
import torch.nn as nn
import numpy as np
import antspynet
from antstorch.architectures import create_unet_model_3d as create_unet_model_3d_pt
import tensorflow as tf
from tensorflow.keras import layers, Model

# Importation de vos fonctions utilitaires existantes pour le fuzzy-matching
import convert_antspynet_weights_to_antstorch as cvt

def transfer_unet_weights(k_unet, p_unet_prefix, sd, number_of_outputs):
    print(f"    -> Mappage des poids pour {p_unet_prefix}...")
    
    # Extraction Keras
    conv_list, dect_list, main_out, _ = cvt._collect_keras_convs_and_dect(k_unet, exclude_layer_names=[])
    
    # Extraction Torch (uniquement le sous-réseau ciblé)
    view_sd = {k: v for k, v in sd.items() if k.startswith(p_unet_prefix)}
    
    t_dect = cvt._list_deconv_keys(view_sd)
    t_enc, t_dec = cvt._list_conv_keys(view_sd)
    t_conv = t_enc + t_dec
    
    # 1. Map Deconvs
    for (lname, kw, kb), tkey in zip(dect_list, t_dect):
        torchW = cvt._to_torch_conv_weight_from_keras(kw)
        rk = cvt._resolve_tkey(tkey, sd)
        if torchW.shape != sd[rk].shape and torchW.transpose(1,0,2,3,4).shape == sd[rk].shape:
            torchW = torchW.transpose(1,0,2,3,4)
        sd[rk] = torch.from_numpy(torchW).to(sd[rk].dtype)
        if kb is not None:
            bkey = cvt._resolve_tkey(tkey.replace(".weight", ".bias"), sd)
            if bkey in sd:
                sd[bkey] = torch.from_numpy(kb.astype(np.float32)).to(sd[bkey].dtype)
                
    # 2. Map Convs (Encoding puis Decoding)
    for (lname, kw, kb), tkey in zip(conv_list, t_conv):
        torchW = cvt._to_torch_conv_weight_from_keras(kw)
        rk = cvt._resolve_tkey(tkey, sd)
        if torchW.shape != sd[rk].shape and torchW.transpose(1,0,2,3,4).shape == sd[rk].shape:
            torchW = torchW.transpose(1,0,2,3,4)
        sd[rk] = torch.from_numpy(torchW).to(sd[rk].dtype)
        if kb is not None:
            bkey = cvt._resolve_tkey(tkey.replace(".weight", ".bias"), sd)
            if bkey in sd:
                sd[bkey] = torch.from_numpy(kb.astype(np.float32)).to(sd[bkey].dtype)
                
    # 3. Map Main Output (Conv 1x1x1 finale)
    lname, kw, kb = main_out
    out_wkey, out_bkey = cvt._find_main_out_keys(view_sd, number_of_outputs)
    torchW = cvt._to_torch_conv_weight_from_keras(kw)
    rk = cvt._resolve_tkey(out_wkey, sd)
    if torchW.shape != sd[rk].shape and torchW.transpose(1,0,2,3,4).shape == sd[rk].shape:
        torchW = torchW.transpose(1,0,2,3,4)
    sd[rk] = torch.from_numpy(torchW).to(sd[rk].dtype)
    if kb is not None and out_bkey:
        rb = cvt._resolve_tkey(out_bkey, sd)
        if rb in sd:
            sd[rb] = torch.from_numpy(kb.astype(np.float32)).to(sd[rb].dtype)


# ==============================================================================
# Classes PyTorch
# ==============================================================================
class CascadeNet(nn.Module):
    def __init__(self, unet0, unet1, concat_inputs=True):
        super().__init__()
        self.unet0 = unet0
        self.unet1 = unet1
        self.concat_inputs = concat_inputs

    def forward(self, x):
        out0 = self.unet0(x)
        if isinstance(out0, (list, tuple)): out0 = out0[0]
        
        if self.concat_inputs:
            # CIT168: On concatène juste le canal T1 (index 0) avec out0
            next_in = torch.cat([x[:, 0:1, ...], out0], dim=1)
        else:
            # NBM: On concatène toutes les entrées avec out0
            next_in = torch.cat([x, out0], dim=1)
            
        out1 = self.unet1(next_in)
        if isinstance(out1, (list, tuple)): out1 = out1[0]
        return out1, out0


# ==============================================================================
# Fonctions de conversion spécifiques
# ==============================================================================
def convert_nbm():
    print("\n--- Début de la conversion pour deep_nbm_rank ---")
    keras_weights_path = antspynet.get_pretrained_network("deep_nbm_rank")
    
    # 1. Construction Keras
    k_unet0 = antspynet.create_unet_model_3d([None, None, None, 1],
        number_of_outputs=1, number_of_filters=(32, 64, 128, 256),
        convolution_kernel_size=3, deconvolution_kernel_size=2,
        pool_size=2, strides=2, dropout_rate=0.0, mode="sigmoid",
        additional_options="nnUnetActivationStyle")
    
    k_unet1 = antspynet.create_unet_model_3d([None, None, None, 2],
        number_of_outputs=9, number_of_filters=(32, 64, 96, 128, 256),
        convolution_kernel_size=3, deconvolution_kernel_size=2,
        pool_size=2, strides=2, dropout_rate=0.0, mode="classification",
        additional_options="nnUnetActivationStyle")
    
    nextin = layers.Concatenate(axis=-1)([k_unet0.inputs[0], k_unet0.outputs[0]])
    k_cascade = Model(inputs=k_unet0.inputs, outputs=[k_unet1(nextin), k_unet0.outputs[0]])
    k_cascade.load_weights(keras_weights_path)

    # 2. Construction PyTorch
    p_unet0 = create_unet_model_3d_pt(input_channel_size=1,
        number_of_outputs=1, number_of_filters=(32, 64, 128, 256),
        convolution_kernel_size=(3,3,3), deconvolution_kernel_size=(2,2,2),
        pool_size=(2,2,2), strides=(2,2,2), dropout_rate=0.0, mode="sigmoid",
        additional_options="nnUnetActivationStyle")

    p_unet1 = create_unet_model_3d_pt(input_channel_size=2,
        number_of_outputs=9, number_of_filters=(32, 64, 96, 128, 256),
        convolution_kernel_size=(3,3,3), deconvolution_kernel_size=(2,2,2),
        pool_size=(2,2,2), strides=(2,2,2), dropout_rate=0.0, mode="classification",
        additional_options="nnUnetActivationStyle")

    p_cascade = CascadeNet(p_unet0, p_unet1, concat_inputs=False)
    
    # 3. Transfert
    sd = p_cascade.state_dict()
    transfer_unet_weights(k_unet0, "unet0.", sd, number_of_outputs=1)
    transfer_unet_weights(k_unet1, "unet1.", sd, number_of_outputs=9)
    p_cascade.load_state_dict(sd)
    
    # 4. Sauvegarde
    out_file = os.path.expanduser("~/.antstorch/deep_nbm_rank_pytorch.pt")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    torch.save(p_cascade.state_dict(), out_file)
    print(f"✓ Sauvegardé avec succès : {out_file}")


def convert_cit(sn_variant=False):
    task_name = "deepCIT168_sn" if sn_variant else "deepCIT168"
    print(f"\n--- Début de la conversion pour {task_name} ---")
    keras_weights_path = antspynet.get_pretrained_network(task_name)
    
    channels = 9
    
    # 1. Construction Keras
    k_unet0 = antspynet.create_unet_model_3d([None, None, None, channels],
        number_of_outputs=1, number_of_filters=(32, 64, 128, 256),
        convolution_kernel_size=3, deconvolution_kernel_size=2,
        pool_size=2, strides=2, dropout_rate=0.0, mode="sigmoid",
        additional_options="nnUnetActivationStyle")
    
    k_unet1 = antspynet.create_unet_model_3d([None, None, None, 2],
        number_of_outputs=9, number_of_filters=(32, 64, 96, 128, 256),
        convolution_kernel_size=3, deconvolution_kernel_size=2,
        pool_size=2, strides=2, dropout_rate=0.0, mode="classification",
        additional_options="nnUnetActivationStyle")
    
    # Utilisation du découpage (slicing) natif de Keras pour isoler le 1er canal
    first_channel = k_unet0.inputs[0][..., 0:1]
    newmult = layers.Concatenate(axis=-1)([first_channel, k_unet0.outputs[0]])
    k_cascade = Model(inputs=k_unet0.inputs, outputs=[k_unet1(newmult), k_unet0.outputs[0]])
    k_cascade.load_weights(keras_weights_path)

    # 2. Construction PyTorch
    p_unet0 = create_unet_model_3d_pt(input_channel_size=channels,
        number_of_outputs=1, number_of_filters=(32, 64, 128, 256),
        convolution_kernel_size=(3,3,3), deconvolution_kernel_size=(2,2,2),
        pool_size=(2,2,2), strides=(2,2,2), dropout_rate=0.0, mode="sigmoid",
        additional_options="nnUnetActivationStyle")

    p_unet1 = create_unet_model_3d_pt(input_channel_size=2,
        number_of_outputs=9, number_of_filters=(32, 64, 96, 128, 256),
        convolution_kernel_size=(3,3,3), deconvolution_kernel_size=(2,2,2),
        pool_size=(2,2,2), strides=(2,2,2), dropout_rate=0.0, mode="classification",
        additional_options="nnUnetActivationStyle")

    p_cascade = CascadeNet(p_unet0, p_unet1, concat_inputs=True)

    # 3. Transfert
    sd = p_cascade.state_dict()
    transfer_unet_weights(k_unet0, "unet0.", sd, number_of_outputs=1)
    transfer_unet_weights(k_unet1, "unet1.", sd, number_of_outputs=9)
    p_cascade.load_state_dict(sd)

    # 4. Sauvegarde
    out_file = os.path.expanduser(f"~/.antstorch/{task_name}_pytorch.pt")
    torch.save(p_cascade.state_dict(), out_file)
    print(f"✓ Sauvegardé avec succès : {out_file}")

if __name__ == "__main__":
    # Conversion des 3 modèles en cascade existants
    convert_nbm()
    convert_cit(sn_variant=False)
    convert_cit(sn_variant=True)
    print("\n✅ Toutes les conversions en cascade sont terminées.")