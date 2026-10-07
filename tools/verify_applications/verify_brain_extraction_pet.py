import os
os.environ['ITK_DEFAULT_GLOBAL_NUMBER_OF_THREADS']='4'
import json, time
from pathlib import Path
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_intra_op_parallelism_threads(4)
import torch
torch.set_num_threads(4)
import ants, antspynet, antstorch
import argparse
parser = argparse.ArgumentParser(description='Compare PET brain extraction with ANTsPyNet on CPU.')
parser.add_argument('--image', required=True)
parser.add_argument('--report', required=True)
args = parser.parse_args()
import numpy as np
image=ants.image_read(args.image)
print(antspynet.__file__, antstorch.__file__, flush=True)
t=time.perf_counter(); a=antspynet.brain_extraction(image,modality='pet',verbose=True); ta=time.perf_counter()-t
print('ANTsPyNet done',ta,flush=True)
t=time.perf_counter(); b=antstorch.brain_extraction(image,modality='pet',device='cpu',verbose=True); tb=time.perf_counter()-t
x=a.numpy(); y=b.numpy(); d=np.abs(x-y); m=x>=.5; n=y>=.5
report=dict(mae=float(d.mean()),max_abs=float(d.max()),dice_threshold_05=float(2*(m&n).sum()/(m.sum()+n.sum())),finite=bool(np.isfinite(x).all() and np.isfinite(y).all()),shape=list(x.shape),antspynet_seconds=ta,antstorch_seconds=tb)
print(json.dumps(report,indent=2),flush=True)
Path(args.report).write_text(json.dumps(report,indent=2)+'\n')
assert report['finite'] and report['max_abs']<1e-3 and report['dice_threshold_05']>.999
