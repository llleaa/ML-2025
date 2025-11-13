import ipywidgets as widgets
from IPython.display import HTML, display

import skimage
import torch
# Import
import os, time, shutil, random, warnings, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, util
#import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

# Avoid warnings
warnings.filterwarnings("ignore", message=".*is a low contrast image", category=UserWarning)
random.seed(1234)
# -----------------------------------------------------
# GUI Tool: select folder
# -----------------------------------------------------
def select_folder(data_path, dataset_name, name='Dataset'):
    folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    folder_selector = widgets.Combobox(
        options=folders, description=name, placeholder=f'Type or select a {name} folder',
        ensure_option=True, disabled=False,
    )
    folder_selector.style.description_width = '90px'
    right_comment = widgets.HTML(value="&nbsp;💡 <i>Select a dataset folder.</i>", layout=widgets.Layout(margin='0 0 0 10px'))
    row = widgets.HBox([folder_selector, right_comment])

    def on_folder_selected(change):
        if change['name'] == 'value' and change['type'] == 'change':
            global dataset_name
            dataset_name = change['new']
            right_comment.value = f"&nbsp;✅ Selected: <b>{dataset_name}</b>"
            
    folder_selector.observe(on_folder_selected, names='value')
    display(row)

# -----------------------------------------------------
# Simulation: create a dataset of randon squares
# -----------------------------------------------------    
def simulate_dataset_squares(project, image_size, type_name, n_imgs, class_properties: dict):
    """
    Create a dummy dataset with square blobs per class and save PNG images/masks.
    project: root directory for dataset
    image_size: width==height in pixelscounts per split)
    num_classes: total classes (0..num_classes-1)
    class_properties: per-class intensity params: {c: {"mean": float, "std": float}}
    """
    num_squares_per_class = 10
    num_classes = len(class_properties)
    square_size_range: tuple = (image_size//12, image_size//8)
    imgd = os.path.join(project, type_name, "images")
    maskd = os.path.join(project, type_name, "masks")
    if os.path.exists(imgd): shutil.rmtree(imgd)
    if os.path.exists(maskd): shutil.rmtree(maskd)
    os.makedirs(imgd, exist_ok=True)
    os.makedirs(maskd, exist_ok=True)

    print(f"Generating {n_imgs} images in {imgd} and {maskd}")
    for i in range(n_imgs):
        image = np.zeros((image_size, image_size), dtype=np.float32)
        mask  = np.zeros((image_size, image_size), dtype=np.uint8)
        for class_id in range(num_classes):
            mean_val = float(class_properties[class_id]["mean"])
            std_val  = float(class_properties[class_id]["std"])
            for _ in range(num_squares_per_class):
                size = random.randint(square_size_range[0], square_size_range[1])
                x = random.randint(0, image_size - size)
                y = random.randint(0, image_size - size)
                square = np.random.normal(mean_val, std_val, (size, size))
                square = np.clip(square, 0, 255)
                image[y:y+size, x:x+size] = square
                mask[y:y+size, x:x+size]  = class_id  # last draw wins (overlap allowed)

        image_u8 = util.img_as_ubyte(np.clip(image, 0, 255) / 255.0) # scale to [0,1] then uint8
        mask_u8  = mask.astype(np.uint8)
        io.imsave(os.path.join(imgd,  f"{i:04d}.png"), image_u8)
        io.imsave(os.path.join(maskd, f"{i:04d}.png"), mask_u8)
        
# -----------------------------------------------------       
# Define a custom Dataset class for loading images and masks
# -----------------------------------------------------

class SegmentationDataset(Dataset):
    def __init__(self, dataset_root, image_subdir="images", mask_subdir="masks"):
        exts=("png","jpg","jpeg","tif","tiff","bmp")
        self.image_dir = os.path.join(dataset_root, image_subdir)
        self.mask_dir  = os.path.join(dataset_root, mask_subdir)
        self.grayscale = True
        exts = tuple(e.lower().lstrip('.') for e in exts)

        def keep(fname):
            return os.path.isfile(os.path.join(self.image_dir, fname)) and os.path.splitext(fname)[1].lower().lstrip('.') in exts

        def keep_m(fname):
            return os.path.isfile(os.path.join(self.mask_dir, fname)) and os.path.splitext(fname)[1].lower().lstrip('.') in exts

        imgs = sorted([f for f in os.listdir(self.image_dir) if keep(f)])
        msks = sorted([f for f in os.listdir(self.mask_dir)  if keep_m(f)])
        img_stems = {os.path.splitext(f)[0].lower(): f for f in imgs}
        msk_stems = {os.path.splitext(f)[0].lower(): f for f in msks}
        img_by_stem = {os.path.splitext(f)[0].lower(): f for f in imgs}
        msk_by_stem = {os.path.splitext(f)[0].lower(): f for f in msks}
        missing_images = sorted(s for s in img_stems.keys() - msk_stems.keys())
        missing_masks  = sorted(s for s in msk_stems.keys() - img_stems.keys())
        if missing_masks : print("Missing masks: ", missing_masks)
        if missing_images: print("Missing images: ", missing_images)
        # Keep only strictly matched pairs (intersection)
        common = sorted(img_stems.keys() & msk_stems.keys())
        self.image_filenames = [img_by_stem[s] for s in common]
        self.mask_filenames  = [msk_by_stem[s] for s in common]
        
    def _quantile_minmax(self, img_np: np.ndarray, low_q=0.01, high_q=0.98) -> np.ndarray:
        x = img_np.astype(np.float32, copy=True)
        lo = np.quantile(x, low_q)
        hi = np.quantile(x, high_q)
        x = np.clip(x, lo, hi)
        x = (x - lo) / (hi - lo)
        return x
        
    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])
        image = io.imread(img_path, as_gray=True).astype(np.float32)
        image = self._quantile_minmax(image)
        mask = io.imread(mask_path, as_gray=True)
        image = torch.from_numpy(image).unsqueeze(0).float() # Add channel dimension and convert to float
        mask = torch.from_numpy(mask).long() # Convert to Long tensor for CrossEntropyLoss
        return image, mask

class SegmentationDataset1(Dataset):
    def __init__(self, dataset_name):
        self.image_dir = f'{dataset_name}/images'
        print(self.image_dir)
        self.mask_dir = f'{dataset_name}/masks'
        self.transform = transforms.Compose([transforms.Normalize(mean=[0.5], std=[0.5])])
        self.image_filenames = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])
        self.mask_filenames = sorted([f for f in os.listdir(self.mask_dir) if f.endswith('.png')])
        print(self.image_filenames)
   
    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])
        image = io.imread(img_path, as_gray=True)
        mask = io.imread(mask_path, as_gray=True)
        image = torch.from_numpy(image).unsqueeze(0).float() # Add channel dimension 
        mask = torch.from_numpy(mask).long() # Convert to Long tensor for CrossEntropyLoss
        if self.transform: image = self.transform(image)
        return image, mask

# -----------------------------------------------------       
# Visualize random raw images and masks
# -----------------------------------------------------
# -----------------------------------------------------       
# Visualize random raw images and masks
# -----------------------------------------------------
def visualization_images(dataset, num_images_to_visualize=3, num_classes=4):
    num_images = min(num_images_to_visualize, len(dataset))
    plt.figure(figsize=(9, num_images * 3))
    indices_to_visualize = random.sample(range(len(dataset)), num_images)
    for i, idx in enumerate(indices_to_visualize):
        image, mask = dataset[idx]
        cmap = plt.get_cmap('viridis', num_classes)
        colored_mask = cmap(mask)[:, :, :3]
        colored_mask = (colored_mask * 255).astype(np.uint8)
        # Plot the image
        plt.subplot(num_images_to_visualize, 2, 2 * i + 1)
        plt.imshow(image[0], cmap='gray')
        plt.title(f"Image: {dataset.image_filenames[idx]}")
        plt.axis('off')
        # Plot the mask
        plt.subplot(num_images_to_visualize, 2, 2 * i + 2)
        plt.imshow(colored_mask) # Use colored_mask
        plt.title(f"Mask: {dataset.mask_filenames[idx]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# -----------------------------------------------------       
# Visualize random raw images, ground-truth and prediction
# -----------------------------------------------------
def visualize_prediction(model, subset, device, sample_idx=0):
    model.eval()
    with torch.no_grad():
        img, msk = subset[sample_idx] 
        inp = img.unsqueeze(0).to(device) 
        logits = model(inp) 
        C = logits.shape[1]
        if C == 1:
            p1 = torch.sigmoid(logits)
            probs = torch.cat([1.0 - p1, p1], dim=1)
            K = 2
        else:
            probs = torch.softmax(logits, dim=1)
            K = C

        pred = probs.argmax(dim=1).squeeze(0).cpu().numpy()

        # Prepare raw grayscale image for display
        img_show = img.squeeze(0).cpu()
        msk_np = msk.cpu().numpy()
        K = max(K, int(msk_np.max()) + 1, int(pred.max()) + 1)

        cmap = plt.get_cmap('viridis', K)
        gt_rgb   = (cmap(msk_np)[:, :, :3] * 255).astype(np.uint8)
        pred_rgb = (cmap(pred)[:,   :, :3] * 255).astype(np.uint8)

        # Plot
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1); plt.imshow(img_show, cmap='gray'); plt.title(f"Image [{sample_idx}]"); plt.axis('off')
        plt.subplot(1, 3, 2); plt.imshow(gt_rgb);  plt.title("Ground Truth");  plt.axis('off')
        plt.subplot(1, 3, 3); plt.imshow(pred_rgb);plt.title("Prediction");    plt.axis('off')
        plt.show()
  
# -----------------------------------------------------       
# Create a table with statistical information
# -----------------------------------------------------
def display_stats(dataset, num_classes=4):
    data = []
    for i, filename in enumerate(dataset.image_filenames):
        image, mask = dataset[i]
        image = image[0].numpy()
        unique_classes, counts = np.unique(mask, return_counts=True)
        class_pixel_counts = dict(zip(unique_classes, counts))
        row_data = [filename, image.shape, np.min(image), np.max(image), np.mean(image), np.std(image)]
        for class_id in range(num_classes): # Assuming class IDs are 0 to num_classes - 1
            row_data.append(class_pixel_counts.get(class_id, 0))
        data.append(row_data)
    columns = ['Filename', 'Shape', 'Min Value', 'Max Value', 'Mean Value', 'Std Deviation']
    for class_id in range(num_classes): columns.append(f'Class {class_id} Pixel Count')
    df_stats = pd.DataFrame(data, columns=columns)
    display(df_stats)

display(HTML("""<style>.message { color:#b00000; background:#ffecec; border:1px solid #ffb3b3; padding:8px; border-radius:4px;}</style>"""))

print(f"Define several core functions")