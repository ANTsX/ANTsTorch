#!/usr/bin/env python3
import os
import re
import torch
import torch.nn as nn
import numpy as np
import antspynet
import tensorflow as tf
from tensorflow.keras import layers, Model
from antstorch.architectures import create_unet_model_3d as create_unet_model_3d_pt

def get_stage_layer_sub(k):
    if 'encoding_convolution_layers' in k:
        m = re.search(r'encoding_convolution_layers\.(\d+)\.(\d+)', k)
        return (0, int(m.group(1)) if m else 0, int(m.group(2)) if m else 0)
    elif 'deconv_activations' in k:
        m = re.search(r'deconv_activations\.(\d+)', k)
        return (1, int(m.group(1)) if m else 0, 0)
    elif 'decoding_convolution_transpose_layers' in k:
        m = re.search(r'decoding_convolution_transpose_layers\.(\d+)', k)
        return (1, int(m.group(1)) if m else 0, -1)
    elif 'decoding_convolution_layers' in k:
        m = re.search(r'decoding_convolution_layers\.(\d+)\.(\d+)', k)
        # CORRECTION : + 1 pour garantir que decoding suit toujours deconv_activations
        return (1, int(m.group(1)) if m else 0, int(m.group(2)) + 1 if m else 1)
    elif 'output' in k or 'head' in k:
        return (2, 0, 0)
    return (99, 0, 0)

def smart_transfer(k_model, p_model):
    sd = p_model.state_dict()
    
    k_convs, k_deconvs, k_norms = [], [], []
    for layer in k_model.layers:
        if isinstance(layer, tf.keras.layers.Conv3D):
            w = layer.get_weights()
            if len(w) > 0: k_convs.append(w)
        elif isinstance(layer, tf.keras.layers.Conv3DTranspose):
            w = layer.get_weights()
            if len(w) > 0: k_deconvs.append(w)
        elif 'norm' in layer.__class__.__name__.lower():
            w = layer.get_weights()
            if len(w) > 0: k_norms.append(w)
            
    p_all_convs = [k for k, v in sd.items() if 'weight' in k and v.ndim == 5]
    p_deconvs_keys = sorted([k for k in p_all_convs if 'transpose' in k.lower()], key=get_stage_layer_sub)
    p_other_convs = sorted([k for k in p_all_convs if 'transpose' not in k.lower()], key=get_stage_layer_sub)
    
    p_all_norms = [name + '.weight' for name, m in p_model.named_modules() if 'ANTsPyNetInstanceNorm' in m.__class__.__name__]
    p_norm_keys_ordered = sorted(p_all_norms, key=get_stage_layer_sub)

    print(f"      [Info] Convolutions: Keras {len(k_convs)} | PyTorch {len(p_other_convs)}")
    print(f"      [Info] Deconvolutions: Keras {len(k_deconvs)} | PyTorch {len(p_deconvs_keys)}")
    print(f"      [Info] Normalisations: Keras {len(k_norms)} | PyTorch {len(p_norm_keys_ordered)}")
    
    success = 0
    def assign(k_w_list, p_w_key, is_conv=True):
        nonlocal success
        kW = k_w_list[0]
        kB = k_w_list[1] if len(k_w_list) > 1 else None
        pW = sd[p_w_key]
        
        if is_conv:
            torchW = np.transpose(kW, (4, 3, 0, 1, 2))
            if torchW.shape != tuple(pW.shape):
                torchW = np.transpose(kW, (3, 4, 0, 1, 2))
        else:
            torchW = kW
            
        torchW = torch.from_numpy(torchW).to(pW.dtype)
        if torchW.shape != pW.shape and torchW.numel() == 1:
            torchW = torchW.expand(pW.shape)
            
        if torchW.shape == tuple(pW.shape):
            sd[p_w_key].copy_(torchW)
            success += 1
        else:
            print(f"      [X] Echec forme W: {torchW.shape} != {pW.shape} pour {p_w_key}")
            
        if kB is not None:
            b_key = p_w_key.replace('.weight', '.bias')
            if b_key in sd:
                b = torch.from_numpy(kB).to(sd[b_key].dtype)
                if b.shape != sd[b_key].shape and b.numel() == 1:
                    b = b.expand(sd[b_key].shape)
                sd[b_key].copy_(b)
                success += 1
                
    for kw, pk in zip(k_convs, p_other_convs): assign(kw, pk, is_conv=True)
    for kw, pk in zip(k_deconvs, p_deconvs_keys): assign(kw, pk, is_conv=True)
    for kw, pk in zip(k_norms, p_norm_keys_ordered): assign(kw, pk, is_conv=False)
        
    p_model.load_state_dict(sd)
    print(f"      ✓ Transféré avec succès : {success} matrices")

class CascadeNet(nn.Module):
    def __init__(self, unet0, unet1):
        super().__init__()
        self.unet0 = unet0
        self.unet1 = unet1

def convert_nbm():
    print("\n--- Début de la conversion pour deep_nbm_rank ---")
    keras_weights_path = antspynet.get_pretrained_network("deep_nbm_rank")
    k_unet0 = antspynet.create_unet_model_3d([None, None, None, 1], number_of_outputs=1, number_of_filters=(32, 64, 128, 256), convolution_kernel_size=3, deconvolution_kernel_size=2, pool_size=2, strides=2, mode="sigmoid", additional_options="nnUnetActivationStyle")
    k_unet1 = antspynet.create_unet_model_3d([None, None, None, 2], number_of_outputs=9, number_of_filters=(32, 64, 96, 128, 256), convolution_kernel_size=3, deconvolution_kernel_size=2, pool_size=2, strides=2, mode="classification", additional_options="nnUnetActivationStyle")
    nextin = layers.Concatenate(axis=-1)([k_unet0.inputs[0], k_unet0.outputs[0]])
    k_cascade = Model(inputs=k_unet0.inputs, outputs=[k_unet1(nextin), k_unet0.outputs[0]])
    k_cascade.load_weights(keras_weights_path)

    p_unet0 = create_unet_model_3d_pt(input_channel_size=1, number_of_outputs=1, number_of_filters=(32, 64, 128, 256), convolution_kernel_size=(3,3,3), deconvolution_kernel_size=(2,2,2), pool_size=(2,2,2), strides=(2,2,2), mode="sigmoid", additional_options=["nnUnetActivationStyle", "kerasDeconvolutionStyle"])
    p_unet1 = create_unet_model_3d_pt(input_channel_size=2, number_of_outputs=9, number_of_filters=(32, 64, 96, 128, 256), convolution_kernel_size=(3,3,3), deconvolution_kernel_size=(2,2,2), pool_size=(2,2,2), strides=(2,2,2), mode="classification", additional_options=["nnUnetActivationStyle", "kerasDeconvolutionStyle"])

    print("  -> Transfert unet0")
    smart_transfer(k_unet0, p_unet0)
    print("  -> Transfert unet1")
    smart_transfer(k_unet1, p_unet1)

    p_cascade = CascadeNet(p_unet0, p_unet1)
    out_file = os.path.expanduser("~/.antstorch/deep_nbm_rank_pytorch.pt")
    torch.save(p_cascade.state_dict(), out_file)
    print(f"✓ Sauvegardé : {out_file}")

def convert_cit(sn_variant=False):
    task_name = "deepCIT168_sn" if sn_variant else "deepCIT168"
    print(f"\n--- Début de la conversion pour {task_name} ---")
    keras_weights_path = antspynet.get_pretrained_network(task_name)
    channels = 9
    k_unet0 = antspynet.create_unet_model_3d([None, None, None, channels], number_of_outputs=1, number_of_filters=(32, 64, 128, 256), convolution_kernel_size=3, deconvolution_kernel_size=2, pool_size=2, strides=2, mode="sigmoid", additional_options="nnUnetActivationStyle")
    k_unet1 = antspynet.create_unet_model_3d([None, None, None, 2], number_of_outputs=9, number_of_filters=(32, 64, 96, 128, 256), convolution_kernel_size=3, deconvolution_kernel_size=2, pool_size=2, strides=2, mode="classification", additional_options="nnUnetActivationStyle")
    first_channel = k_unet0.inputs[0][..., 0:1]
    newmult = layers.Concatenate(axis=-1)([first_channel, k_unet0.outputs[0]])
    k_cascade = Model(inputs=k_unet0.inputs, outputs=[k_unet1(newmult), k_unet0.outputs[0]])
    k_cascade.load_weights(keras_weights_path)

    p_unet0 = create_unet_model_3d_pt(input_channel_size=channels, number_of_outputs=1, number_of_filters=(32, 64, 128, 256), convolution_kernel_size=(3,3,3), deconvolution_kernel_size=(2,2,2), pool_size=(2,2,2), strides=(2,2,2), mode="sigmoid", additional_options=["nnUnetActivationStyle", "kerasDeconvolutionStyle"])
    p_unet1 = create_unet_model_3d_pt(input_channel_size=2, number_of_outputs=9, number_of_filters=(32, 64, 96, 128, 256), convolution_kernel_size=(3,3,3), deconvolution_kernel_size=(2,2,2), pool_size=(2,2,2), strides=(2,2,2), mode="classification", additional_options=["nnUnetActivationStyle", "kerasDeconvolutionStyle"])

    print("  -> Transfert unet0")
    smart_transfer(k_unet0, p_unet0)
    print("  -> Transfert unet1")
    smart_transfer(k_unet1, p_unet1)

    p_cascade = CascadeNet(p_unet0, p_unet1)
    out_file = os.path.expanduser(f"~/.antstorch/{task_name}_pytorch.pt")
    torch.save(p_cascade.state_dict(), out_file)
    print(f"✓ Sauvegardé : {out_file}")

if __name__ == "__main__":
    convert_nbm()
    convert_cit(sn_variant=False)
    convert_cit(sn_variant=True)