"""
Unaligned Dataset with Mask Support
====================================
Paper reference: Section 3.1 Data Preparation

Dataset class for unpaired image translation with optional binary masks.
Loads images from domain A (input features) and domain B (target H&E),
along with pre-computed physical masks from domain A.

Directory structure expected under --dataroot:
- trainA/       : Input feature images (merged FFT + TrueColor channels)
- trainB/       : Target H&E-stained images
- trainA_mask/  : Binary tissue masks (from IPCA+KMeans pipeline)

Key features:
- Filename-stem based alignment between A images and masks
- Synchronized geometric transforms (crop, flip) between image and mask
- Nearest-neighbor interpolation for masks to preserve binary values
- Force binarization after transforms (>0.5 threshold)
"""
import os.path
import random
import torch
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform, get_params
from data.image_folder import make_dataset
from PIL import Image
import util.util as util

class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets with optional masks.

    It requires three directories:
    1. '/path/to/data/trainA': Training images for domain A.
    2. '/path/to/data/trainB': Training images for domain B.
    3. '/path/to/data/trainA_mask': Binary masks for domain A (optional, controlled by --use_mask).

    The filenames in 'trainA' and 'trainA_mask' must correspond to each other 
    (matching by filename stem, e.g., 'image_01.tif' matches 'image_01.png').
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        
        # Mask directory for domain A (optional)
        self.dir_A_mask = os.path.join(opt.dataroot, opt.phase + 'A_mask')
        
        # Check if mask directory exists and use_mask flag is set
        self.use_mask = os.path.exists(self.dir_A_mask) and hasattr(opt, 'use_mask') and opt.use_mask

        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")
            if self.use_mask:
                self.dir_A_mask = os.path.join(opt.dataroot, "valA_mask")

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))   # load images from '/path/to/data/trainB'
        
        # Load mask paths if available
        if self.use_mask:
            self.A_mask_paths = self._align_mask_paths(self.A_paths, self.dir_A_mask)
        
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

    def _align_mask_paths(self, A_paths, mask_dir):
        """Align mask paths with A image paths based on filename stems.
        
        It assumes strict consistency as guaranteed by the user. 
        It matches files like 'image_01.tif' in A with 'image_01.png' in Mask.
        """
        aligned_paths = []
        # Get all mask files
        all_masks = sorted(make_dataset(mask_dir, float("inf")))
        
        # Create a mapping: filename_stem -> full_path
        # e.g., 'slide_001' -> '/path/to/mask/slide_001.png'
        mask_map = {os.path.splitext(os.path.basename(p))[0]: p for p in all_masks}

        for a_path in A_paths:
            # Get the stem of the A image
            base_name = os.path.splitext(os.path.basename(a_path))[0]
            
            if base_name in mask_map:
                aligned_paths.append(mask_map[base_name])
            else:
                # If consistency is guaranteed, this theoretically shouldn't happen,
                # but we handle it safely by appending None.
                print(f"Warning: Mask for '{base_name}' not found in '{mask_dir}'.")
                aligned_paths.append(None)
        
        return aligned_paths

    def _get_mask_transform(self, opt, params):
        """Define transform specifically for Mask.
        
        1. Must use Nearest Neighbor interpolation to preserve binary values.
        2. No normalization applied (values remain 0 or 1).
        """
        transform_list = []
        
        # 1. Resize / Crop (Must match Image A geometric parameters)
        if 'resize' in opt.preprocess:
            osize = [opt.load_size, opt.load_size]
            transform_list.append(transforms.Resize(osize, interpolation=transforms.InterpolationMode.NEAREST))
        elif 'scale_width' in opt.preprocess:
            # FIX: Renamed helper function call to avoid name mangling
            transform_list.append(transforms.Lambda(lambda img: scale_width_helper(img, opt.load_size, opt.crop_size, method=transforms.InterpolationMode.NEAREST)))

        if 'crop' in opt.preprocess:
            if params is None:
                transform_list.append(transforms.RandomCrop(opt.crop_size))
            else:
                # FIX: Renamed helper function call to avoid name mangling
                transform_list.append(transforms.Lambda(lambda img: crop_helper(img, params['crop_pos'], opt.crop_size)))

        if opt.preprocess == 'none':
            # FIX: Renamed helper function call
            transform_list.append(transforms.Lambda(lambda img: make_power_2_helper(img, base=4, method=transforms.InterpolationMode.NEAREST)))

        # 2. Flip (Geometric consistency)
        if not opt.no_flip:
            if params is None:
                transform_list.append(transforms.RandomHorizontalFlip())
            elif params['flip']:
                # FIX: Renamed helper function call
                transform_list.append(transforms.Lambda(lambda img: flip_helper(img, params['flip'])))

        # 3. ToTensor
        transform_list.append(transforms.ToTensor())
        
        return transforms.Compose(transform_list)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_mask (tensor)  -- mask for image A (if available)
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within the range
        if self.opt.serial_batches:   # make sure index is within the range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        
        # Load images
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        
        # Load Mask (if available)
        A_mask = None
        if self.use_mask:
            mask_path = self.A_mask_paths[index % self.A_size]
            if mask_path is not None:
                try:
                    A_mask = Image.open(mask_path).convert('L')
                except Exception as e:
                    print(f"Error loading mask {mask_path}: {e}")
                    A_mask = Image.new('L', A_img.size, 0) # Fallback to black mask
            else:
                A_mask = Image.new('L', A_img.size, 0) # No mask found

        # Apply image transformation
        # For CUT/FastCUT mode, if in finetuning phase (learning rate is decaying),
        # do not perform resize-crop data augmentation of CycleGAN.
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        
        # Core: get geometric transformation parameters to synchronize Image and Mask
        params = get_params(modified_opt, A_img.size)
        
        # Get transforms for images
        transform_A = get_transform(modified_opt, params=params, grayscale=(self.opt.input_nc == 1))
        transform_B = get_transform(modified_opt) # B is unpaired, no sync needed
        
        # Apply transforms to images
        A = transform_A(A_img)
        B = transform_B(B_img)
        
        result = {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

        # Apply transforms to mask
        if A_mask is not None:
            # Use custom mask transform with synced params
            transform_mask = self._get_mask_transform(modified_opt, params)
            A_mask_tensor = transform_mask(A_mask)
            
            # [FIX] Force Binarization: Convert any interpolated values to strict 0.0 or 1.0
            # This is critical because ToTensor() scales [0,255] to [0,1], 
            # and we need to ensure values are not like 0.0039 (1/255).
            A_mask_tensor = (A_mask_tensor > 0.5).float()
            
            # Ensure mask is single channel [1, H, W]
            if A_mask_tensor.shape[0] > 1:
                A_mask_tensor = A_mask_tensor[0:1, ...]
                
            result['A_mask'] = A_mask_tensor

        return result

    def __len__(self):
        """Return the total number of images in the dataset."""
        return max(self.A_size, self.B_size)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add dataset-specific options."""
        parser.add_argument('--use_mask', action='store_true', help='whether to use mask for domain A images')
        return parser


# Helper functions for geometric transforms
# [IMPORTANT FIX]: Renamed functions to remove double underscores (__)
# This prevents Python from treating them as private class members when called inside the class.

def scale_width_helper(img, target_width, crop_width, method=transforms.InterpolationMode.BICUBIC):
    ow, oh = img.size
    if ow == target_width and oh >= crop_width:
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)

def crop_helper(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img

def flip_helper(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def make_power_2_helper(img, base, method=transforms.InterpolationMode.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img
    return img.resize((w, h), method)