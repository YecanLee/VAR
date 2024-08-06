import os
import os.path as osp
import torch 
import torchvision
import random
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
from tqdm import trange
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from models import VQVAE, build_vae_var

MODEL_DEPTH = 16    # TODO: =====> please specify MODEL_DEPTH <=====
assert MODEL_DEPTH in {16, 20, 24, 30}


# download checkpoint
hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
vae_ckpt, var_ckpt = 'vae_ch160v4096z32.pth', f'var_d{MODEL_DEPTH}.pth'
if not osp.exists(vae_ckpt): os.system(f'wget {hf_home}/{vae_ckpt}')
if not osp.exists(var_ckpt): os.system(f'wget {hf_home}/{var_ckpt}')

# build vae, var
patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if 'vae' not in globals() or 'var' not in globals():
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
    )

# load checkpoints
vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
vae.eval(), var.eval()
for p in vae.parameters(): p.requires_grad_(False)
for p in var.parameters(): p.requires_grad_(False)
print(f'prepare finished.')


# set args
num_sampling_steps = 250  #@param {type:"slider", min:0, max:1000, step:1}
cfg = 2  #@param {type:"slider", min:1, max:10, step:0.1}
class_labels = tuple(range(1000))  #@param {type:"raw"}
more_smooth = False  # True for more smooth output
batch_size = 2400  # Specify the batch size for generating images
num_images_per_class = 50  # Specify the number of images to generate for each class

# Create the output folder if it doesn't exist
output_folder = 'generated_images'
os.makedirs(output_folder, exist_ok=True)

# run faster
tf32 = True
torch.backends.cudnn.allow_tf32 = bool(tf32)
torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
torch.set_float32_matmul_precision('high' if tf32 else 'highest')

# sample
num_batches = (len(class_labels) + batch_size - 1) // batch_size

with torch.inference_mode(): # use inference mode to avoid storing intermediate activations
    with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True): # use autocast to speed up inference
        for class_idx in trange(len(class_labels)):
            for image_idx in range(num_images_per_class):
                # Generate a new random seed for each image, otherwise the generated images will be the same
                seed = random.randint(0, 2**32 - 1)
                
                # Set the random seed
                torch.manual_seed(seed)
                random.seed(seed)
                np.random.seed(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                
                label_B: torch.LongTensor = torch.tensor([class_labels[class_idx]], device=device)
                
                recon_B3HW = var.autoregressive_infer_cfg(B=1, label_B=label_B, cfg=cfg, top_k=900, top_p=0.95, g_seed=seed, more_smooth=more_smooth)
                
                # Save the generated image
                chw = torchvision.utils.make_grid(recon_B3HW, nrow=1, padding=0, pad_value=1.0)
                chw = chw.permute(1, 2, 0).mul_(255).cpu().numpy()
                chw = PImage.fromarray(chw.astype(np.uint8))
                filename = f'class_{class_idx}_image_{image_idx}.png'
                chw.save(os.path.join(output_folder, filename))