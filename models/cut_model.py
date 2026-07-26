"""
HSCUT Model: Hyperspectral Contrastive Unpaired Translation
============================================================
Paper reference: Section 2.2, Methods Section 3.2

Core model implementing CUT/FastCUT with K-Means-based one-way
inclusion mask loss (loss_MASK).

Architecture:
- Generator G (NetG): Translates input features to virtual H&E
- MLP Projector F (NetF): Patch-wise feature sampling for contrastive learning
- Discriminator D (NetD): Adversarial discriminator

Losses (Eq. in Section 3.2):
- L_GAN: Adversarial loss ensuring output realism
- L_NCE: Patch-wise contrastive loss maximizing input-output mutual information
- L_MASK: One-way inclusion constraint anchoring true voids as background
         (only penalizes background->tissue misclassification, not vice versa)

Key options:
--lambda_mask: Weight for mask consistency loss (>0 to enable)
--kmeans_path: Path to pre-trained K-Means model on target domain
--bg_label:    K-Means cluster label index for background class
"""
import numpy as np
import torch
import os
import joblib
import pickle
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util


class CUTModel(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss: GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        
        # --- NEW: K-Means Mask Loss Options ---
        parser.add_argument('--lambda_mask', type=float, default=0.0, help='weight for mask consistency loss. Set > 0.0 to enable.')
        parser.add_argument('--kmeans_path', type=str, default='./k_means/kmeans_model.pkl', help='path to the pre-trained sklearn kmeans model')
        parser.add_argument('--bg_label', type=int, default=0, help='the cluster label index that corresponds to the background/mask area')
        # --------------------------------------

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        # Add Mask loss to tracking if enabled
        if self.isTrain and opt.lambda_mask > 0.0:
            self.loss_names += ['mask']
            # Optional: Visualize the predicted mask to debug the constraint area
            self.visual_names += ['pred_mask_prob', 'real_A_mask'] 

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            
            # Mask Loss criterion
            # Use 'none' reduction to compute pixel-wise loss, allowing strictly masked application later.
            self.criterionMask = torch.nn.L1Loss(reduction='none').to(self.device)

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            
            # --- NEW: Load Pre-trained K-Means Centroids (Robust Loading) ---
            if self.opt.lambda_mask > 0.0:
                if os.path.exists(self.opt.kmeans_path):
                    print(f"Loading K-Means model from {self.opt.kmeans_path} for Mask Loss...")
                    
                    kmeans_model = None
                    load_success = False
                    
                    # Attempt 1: Try using joblib
                    try:
                        kmeans_model = joblib.load(self.opt.kmeans_path)
                        load_success = True
                        print("Successfully loaded with joblib.")
                    except Exception as e:
                        print(f"Warning: joblib.load failed with error: {e}. Trying standard pickle...")

                    # Attempt 2: Try using standard pickle if joblib failed
                    if not load_success:
                        try:
                            with open(self.opt.kmeans_path, 'rb') as f:
                                kmeans_model = pickle.load(f)
                            load_success = True
                            print("Successfully loaded with pickle.")
                        except Exception as e:
                            print(f"Error: Failed to load K-Means model with both joblib and pickle.")
                            print(f"Please check if '{self.opt.kmeans_path}' is corrupted or a Git LFS pointer (text file).")
                            raise e

                    # Extract cluster centers
                    # Handle different sklearn versions or data structures
                    if hasattr(kmeans_model, 'cluster_centers_'):
                        centroids = kmeans_model.cluster_centers_
                    elif isinstance(kmeans_model, np.ndarray):
                        # In case the file only contains the centroids array
                        centroids = kmeans_model
                        print("Loaded object is a numpy array, assuming it represents centroids.")
                    else:
                        raise ValueError("Loaded object is neither a KMeans model nor a numpy array of centroids.")

                    # Convert to Tensor and move to GPU
                    # NOTE: Ensure your K-Means was trained on data in range [0, 255]
                    self.cluster_centers = torch.from_numpy(centroids).float().to(self.device)
                    
                    # Reshape for broadcasting: (1, n_clusters, 3)
                    if len(self.cluster_centers.shape) == 2:
                        self.cluster_centers = self.cluster_centers.unsqueeze(0)
                    
                    print(f"K-Means initialized. Centroids shape: {self.cluster_centers.shape}")
                else:
                    print(f"WARNING: K-Means model not found at {self.opt.kmeans_path}. Mask loss will be ignored.")
                    self.opt.lambda_mask = 0.0

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        
        # --- NEW: Load Mask if available ---
        # The dataset guarantees 'A_mask' exists if use_mask is True
        if 'A_mask' in input:
            self.real_A_mask = input['A_mask'].to(self.device)
        else:
            self.real_A_mask = None

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        # --- NEW: Compute K-Means Mask Loss (One-Way / Inclusion Constraint) ---
        self.loss_mask = 0.0
        # Only compute if lambda_mask > 0 and a mask exists for the current batch
        if self.opt.lambda_mask > 0.0 and self.real_A_mask is not None:
            # 1. Denormalize fake_B from [-1, 1] to [0, 255]
            # IMPORTANT: This assumes your K-means model was trained on [0, 255] pixel values.
            # If K-means was trained on [0, 1], change 255.0 to 1.0.
            fake_B_denorm = (self.fake_B + 1) / 2.0 * 255.0
            
            # 2. Prepare for clustering: (B, 3, H, W) -> (B, H, W, 3) -> (B*H*W, 3)
            B, C, H, W = self.fake_B.shape
            pixel_values = fake_B_denorm.permute(0, 2, 3, 1).reshape(-1, C) 
            flat_centers = self.cluster_centers.squeeze(0) # (n_clusters, 3)
            
            # 3. Calculate Squared Euclidean Distance to each centroid
            # torch.cdist computes L2 distance (Euclidean). 
            dists = torch.cdist(pixel_values, flat_centers, p=2) # (B*H*W, n_clusters)
            
            # 4. Soft-Assignment (Differentiable)
            # Use negative distance with Softmax. 
            # Temperature T controls sharpness: high T -> uniform, low T -> one-hot.
            temperature = 20.0 
            probs = torch.nn.functional.softmax(-dists / temperature, dim=1) # (B*H*W, n_clusters)
            
            # 5. Extract probability of the Background Label
            # We want this probability to be high where the input mask says "Background"
            bg_prob = probs[:, self.opt.bg_label] # (B*H*W, )
            bg_prob_map = bg_prob.reshape(B, 1, H, W) # (B, 1, H, W)
            
            # For visualization (detach to avoid graph retention)
            self.pred_mask_prob = bg_prob_map.detach()

            # 6. Prepare the Target Mask (Constraint Weight)
            mask_weight = self.real_A_mask
            
            # [FastCUT Fix] Handle Flip Equivariance
            # In FastCUT mode, 'forward()' might randomly flip the input 'real_A'.
            # Consequently, 'fake_B' will be a generated image of the FLIPPED input.
            # We must flip 'mask_weight' (real_A_mask) to match the geometry of 'fake_B'.
            # 'self.flipped_for_equivariance' is set in forward().
            if self.opt.flip_equivariance and getattr(self, 'flipped_for_equivariance', False):
                mask_weight = torch.flip(mask_weight, [3]) # Flip along width dimension (N, C, H, W)

            # 7. Calculate Pixel-wise Loss (Inclusion Logic)
            # Loss = |bg_prob - target_mask|
            # We compare the generated prob map (which might be flipped) 
            # with the appropriately adjusted mask (which is now also flipped if needed).
            pixel_loss = self.criterionMask(bg_prob_map, mask_weight)
            
            # Apply One-Way Inclusion:
            # - Region where Mask=1 (Input is Background): 
            #   We want Output to be Background (prob=1). pixel_loss here is high if prob is low.
            #   masked_loss = pixel_loss * 1. Constraint is ACTIVE.
            #
            # - Region where Mask=0 (Input is Tissue):
            #   We DO NOT care if Output is Background or Tissue. 
            #   pixel_loss here calculates distance to 0, BUT we multiply by 0.
            #   masked_loss = pixel_loss * 0. Constraint is IGNORED.
            masked_loss = pixel_loss * mask_weight
            
            # 8. Normalize by the number of background pixels to avoid dilution
            # We only average over the active constraint area (Background area).
            num_bg_pixels = torch.sum(mask_weight) + 1e-6
            
            self.loss_mask = torch.sum(masked_loss) / num_bg_pixels * self.opt.lambda_mask
        # -------------------------------------

        self.loss_G = self.loss_G_GAN + loss_NCE_both + self.loss_mask
        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers