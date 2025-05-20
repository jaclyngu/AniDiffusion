import sys
# sys.path.insert(0, '/sensei-fs/users/diliu/zeqi_project/AnimateDiff')
import shutil
import argparse
import logging
import math
import os
import os.path as osp
import random
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import imageio
import time
import random
import wandb
os.environ['WANDB_API_KEY'] = 'd2da24233fbf26b5f09692a28dea488df97dd2c5'
print('here')
import diffusers
import mlflow
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torchvision.transforms.functional import to_pil_image
import transformers
print('here1')
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPVisionModelWithProjection
from torch.autograd import Variable
print('here2')
# from src.dwpose import DWposeDetector
from src.models.mutual_self_attention import ReferenceAttentionControl
print('here3')
from src.models.pose_guider import PoseGuider
print('here4')
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
print('here5')
from src.pipelines.pipeline_pose2img import Pose2ImagePipeline
from src.utils.util import delete_additional_ckpt, import_filename, seed_everything
print('here6')
# from animatediff.models.ldm.util import instantiate_from_config

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config, unzip_dict=False):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    if unzip_dict:
        return get_obj_from_str(config["target"])(**config.get("params", dict()))
    else:
        return get_obj_from_str(config["target"])(config.get("params", dict()))

warnings.filterwarnings("ignore")

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")
# import onnxruntime as ort
# ort.set_default_logger_severity(1)


class Net(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        pose_guider: PoseGuider,
        reference_control_writer,
        reference_control_reader,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.pose_guider = pose_guider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents,
        clip_image_embeds,
        pose_img,
        uncond_fwd: bool = False,
    ):
        pose_cond_tensor = pose_img.to(device="cuda")
        pose_fea = self.pose_guider(pose_cond_tensor)

        if not uncond_fwd:
            ref_timesteps = torch.zeros_like(timesteps)
            self.reference_unet(
                ref_image_latents,
                ref_timesteps,
                encoder_hidden_states=clip_image_embeds,
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer)
        # print(noisy_latents.size(), timesteps, clip_image_embeds.size(),  pose_fea.size(), uncond_fwd)
        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            pose_cond_fea=pose_fea,
            encoder_hidden_states=clip_image_embeds,
        ).sample

        return model_pred


def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
        timesteps
    ].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
        device=timesteps.device
    )[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


def log_validation(
    dataloader,
    vae,
    image_enc,
    net,
    scheduler,
    accelerator,
    cfg,
    val_max_img,
):
    logger.info("Running validation... ")
    width=cfg.data.train_width
    height=cfg.data.train_height

    ori_net = accelerator.unwrap_model(net)
    reference_unet = ori_net.reference_unet
    denoising_unet = ori_net.denoising_unet
    pose_guider = ori_net.pose_guider

    # cast unet dtype
    vae = vae.to(dtype=torch.float32)
    image_enc = image_enc.to(dtype=torch.float32)

    # pose_detector = DWposeDetector()
    # pose_detector.to(accelerator.device)

    pipe = Pose2ImagePipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
    )
    pipe = pipe.to(accelerator.device)
    pil_images = []
    val_loss=[]
    for idx, batch in enumerate(dataloader):
        print('in log', idx, batch["ref_img"].size())
        if idx >= val_max_img:
            break

        # ref_image_pil = Image.open("./configs/inference/ref_images/anyone-3.png").convert("RGB")
        # pose_image_pil = Image.open("./configs/inference/pose_images/pose-1.png").convert("RGB")
        batch=get_inputs(batch)
        for i in range(batch["ref_img"].size(0)):
            generator = torch.Generator(device=accelerator.device).manual_seed(42)
            # print(batch["ref_img"][i].size(), batch["tgt_pose"][i].size())
            ref_image_pil = to_pil_image(batch["ref_img"][i])
            pose_image_pil = to_pil_image(batch["tgt_pose"][i])
            gt_image_pil = to_pil_image(batch["img"][i])
            ref_pose_image_pil = to_pil_image(batch["canonical_vis"][i])

            image = pipe(
                ref_image_pil,
                pose_image_pil,
                width,
                height,
                20,
                3.5,
                generator=generator,
            ).images
            #both range 0...1
            image = image[0, :, 0].permute(1, 2, 0).cpu().numpy()  # (3, 512, 512)
            gt_np = batch["img"][i].permute(1, 2, 0).cpu().numpy()
            print(image.max(), image.min(), gt_np.max(), gt_np.min())
            val_loss.append(((image-gt_np)**2).mean())
            res_image_pil = Image.fromarray((image * 255).astype(np.uint8))
            # Save ref_image, src_image and the generated_image
            w, h = res_image_pil.size
            canvas = Image.new("RGB", (w * 4, h), "white")
            ref_image_pil = ref_image_pil.resize((w, h))
            pose_image_pil = pose_image_pil.resize((w, h))
            canvas.paste(ref_image_pil, (0, 0))
            canvas.paste(pose_image_pil, (w, 0))
            # canvas.paste(ref_pose_image_pil, (w*2, 0))
            canvas.paste(gt_image_pil, (w*2, 0))
            canvas.paste(res_image_pil, (w * 3, 0))

            pil_images.append({"name": f"{idx}_{i}", \
                                "img": canvas,\
                                "ref_img":ref_image_pil,\
                                "pose_img": pose_image_pil,\
                                'ref_pose_img':ref_pose_image_pil,\
                                "gt_img":gt_image_pil,\
                                "res_img":res_image_pil})

    vae = vae.to(dtype=torch.float16)
    image_enc = image_enc.to(dtype=torch.float16)

    del pipe
    torch.cuda.empty_cache()
    print('val_loss', val_loss)
    return pil_images, val_loss

def get_inputs(batch):
    for ck in batch:
        if torch.is_tensor(batch[ck]) and len(batch[ck].size())==5:
            b,f,c,h,w=batch[ck].size()
            batch[ck]=batch[ck].view(-1, c,h,w)
    return batch

def main(args, cfg):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    kwargs_time = InitProcessGroupKwargs(timeout=timedelta(seconds=180000000))

    start=time.time()
    use_wandb=args.use_wandb

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        mixed_precision=cfg.solver.mixed_precision,
        log_with="mlflow",
        project_dir="./mlruns",
        kwargs_handlers=[kwargs, kwargs_time],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    print('1', time.time()-start)
    # If passed along, set the training seed now.
    if cfg.seed is not None:
        seed_everything(cfg.seed)

    exp_name = cfg.save_name
    save_dir = f"{cfg.output_dir}/{exp_name}"
    if accelerator.is_main_process and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ########Back up files
    if accelerator.is_main_process:
        shutil.copyfile(args.config, os.path.join(save_dir, 'config.yaml'))
        shutil.copyfile('/'.join(cfg.train_data.target.split('.')[:-1])+'.py', os.path.join(save_dir, 'train_data.py'))
        shutil.copyfile('/'.join(cfg.validation_data.target.split('.')[:-1])+'.py', os.path.join(save_dir, 'validation_data.py'))
        shutil.copyfile(sys.argv[0], os.path.join(save_dir, 'script.py'))
        log_time_dir=save_dir
    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif cfg.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        raise ValueError(
            f"Do not support weight dtype: {cfg.weight_dtype} during training"
        )

    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    start=time.time()
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})
    train_noise_scheduler = DDIMScheduler(**sched_kwargs)
    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(
        "cuda", dtype=weight_dtype
    )

    reference_unet = UNet2DConditionModel.from_pretrained(
        cfg.base_model_path,
        subfolder="unet",
    ).to(device="cuda")
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        cfg.base_model_path,
        "",
        subfolder="unet",
        unet_additional_kwargs={
            "use_motion_module": False,
            "unet_use_temporal_attention": False,
        },
    ).to(device="cuda")

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        cfg.image_encoder_path,
    ).to(dtype=weight_dtype, device="cuda")

    if cfg.pose_guider_pretrain:
        pose_guider = PoseGuider(
            conditioning_embedding_channels=320, block_out_channels=(16, 32, 96, 256)
        ).to(device="cuda")
        # load pretrained controlnet-openpose params for pose_guider
        controlnet_openpose_state_dict = torch.load(cfg.controlnet_openpose_path)
        state_dict_to_load = {}
        for k in controlnet_openpose_state_dict.keys():
            if k.startswith("controlnet_cond_embedding.") and k.find("conv_out") < 0:
                new_k = k.replace("controlnet_cond_embedding.", "")
                state_dict_to_load[new_k] = controlnet_openpose_state_dict[k]
        miss, _ = pose_guider.load_state_dict(state_dict_to_load, strict=False)
        logger.info(f"Missing key for pose guider: {len(miss)}")
    else:
        pose_guider = PoseGuider(
            conditioning_embedding_channels=320, block_out_channels=(16, 32, 96, 256)
        ).to(device="cuda")

    # load pretrained weights
    if cfg.denoising_unet_path !='':
        denoising_unet.load_state_dict(
            torch.load(cfg.denoising_unet_path), # map_location="cpu"),
            strict=False,
        )
    if cfg.reference_unet_path !='':
        reference_unet.load_state_dict(
            torch.load(cfg.reference_unet_path), # map_location="cpu"),
        )
    if cfg.pose_guider_path !='':
        pose_guider.load_state_dict(
            torch.load(cfg.pose_guider_path), # map_location="cpu"),
        )

    # Freeze
    vae.requires_grad_(False)
    image_enc.requires_grad_(False)

    # Explictly declare training models
    denoising_unet.requires_grad_(True)
    #  Some top layer parames of reference_unet don't need grad
    for name, param in reference_unet.named_parameters():
        if "up_blocks.3" in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)

    pose_guider.requires_grad_(True) #just s.t. the optimizer param list is not empty

    reference_control_writer = ReferenceAttentionControl(
        reference_unet,
        do_classifier_free_guidance=False,
        mode="write",
        fusion_blocks="full",
    )
    reference_control_reader = ReferenceAttentionControl(
        denoising_unet,
        do_classifier_free_guidance=False,
        mode="read",
        fusion_blocks="full",
    )

    net = Net(
        reference_unet,
        denoising_unet,
        pose_guider,
        reference_control_writer,
        reference_control_reader,
    )
    print('2', time.time()-start)
    start=time.time()
    if cfg.solver.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            reference_unet.enable_xformers_memory_efficient_attention()
            denoising_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if cfg.solver.gradient_checkpointing:
        reference_unet.enable_gradient_checkpointing()
        denoising_unet.enable_gradient_checkpointing()

    if cfg.solver.scale_lr:
        learning_rate = (
            cfg.solver.learning_rate
            * cfg.solver.gradient_accumulation_steps
            * cfg.data.train_bs
            * accelerator.num_processes
        )
    else:
        learning_rate = cfg.solver.learning_rate

    # Initialize the optimizer
    if cfg.solver.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    print('3', time.time()-start)
    start=time.time()
    trainable_params = list(filter(lambda p: p.requires_grad, net.parameters()))
    optimizer = optimizer_cls(
        trainable_params,
        lr=learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )

    # Scheduler
    lr_scheduler = get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.solver.lr_warmup_steps
        * cfg.solver.gradient_accumulation_steps,
        num_training_steps=cfg.solver.max_train_steps
        * cfg.solver.gradient_accumulation_steps,
    )

    # train_dataset = HumanDanceDataset(
    #     img_size=(cfg.data.train_width, cfg.data.train_height),
    #     img_scale=(0.9, 1.0),
    #     data_meta_paths=cfg.data.meta_paths,
    #     sample_margin=cfg.data.sample_margin,
    # )
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    train_dataset = instantiate_from_config(OmegaConf.create(cfg.train_data), unzip_dict=False)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.data.train_bs, shuffle=True, num_workers=4, drop_last=True
    )

    validation_dataset = instantiate_from_config(OmegaConf.create(cfg.validation_data), unzip_dict=False)
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=cfg.data.validation_bs, shuffle=True if args.mode=='train' else args.shuffle, num_workers=4, drop_last=True
    )

    # Prepare everything with our `accelerator`.
    (
        net,
        optimizer,
        train_dataloader,
        validation_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        net,
        optimizer,
        train_dataloader,
        validation_dataloader,
        lr_scheduler,
    )
    print('4', time.time()-start)
    start=time.time()
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.solver.gradient_accumulation_steps
    )
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(
        cfg.solver.max_train_steps / num_update_steps_per_epoch
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run_time = datetime.now().strftime("%Y%m%d-%H%M")
        accelerator.init_trackers(
            cfg.exp_name,
            init_kwargs={"mlflow": {"run_name": run_time}},
        )
        # dump config file
        mlflow.log_dict(OmegaConf.to_container(cfg), "config.yaml")
        print('max mem after accelerator.prepare',torch.cuda.max_memory_allocated()/1e6)
        # torch.cuda.reset_peak_memory_stats() 

        if use_wandb:
            run = wandb.init(
                config=args,
                project=os.path.basename(os.path.normpath(cfg.output_dir)),
                entity=None,
                name='_'.join([cfg.exp_name, cfg.save_name, run_time]),
                reinit=True
            )

    # Train!
    total_batch_size = (
        cfg.data.train_bs
        * accelerator.num_processes
        * cfg.solver.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.data.train_bs}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {cfg.solver.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {cfg.solver.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            #the format needs to be [epoch_number]_backup
            resume_dir = cfg.resume_from_checkpoint
            accelerator.load_state(resume_dir)
            accelerator.print(f"Resuming from checkpoint {resume_dir}")
            global_step = int(os.path.basename(resume_dir).split("_")[0])
        else:
            resume_dir = f"{cfg.output_dir}/{cfg.exp_name}"
            # Get the most recent checkpoint
            dirs = os.listdir(resume_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1]
            accelerator.load_state(os.path.join(resume_dir, path))
            accelerator.print(f"Resuming from checkpoint {path}")
            global_step = int(path.split("-")[1])

        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, cfg.solver.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")
    print('5', time.time()-start, first_epoch, num_train_epochs, num_update_steps_per_epoch, global_step)
    if args.mode=='train':
        for epoch in range(first_epoch, num_train_epochs):
            train_loss = 0.0
            for step, batch in enumerate(train_dataloader):
                batch=get_inputs(batch)
                with accelerator.accumulate(net):
                    # Convert videos to latent space
                    pixel_values = batch["img"].to(weight_dtype)
                    # print(pixel_values.size())
                    with torch.no_grad():
                        latents = vae.encode(pixel_values).latent_dist.sample()
                        latents = latents.unsqueeze(2)  # (b, c, 1, h, w)
                        latents = latents * 0.18215

                    noise = torch.randn_like(latents)
                    if cfg.noise_offset > 0.0:
                        noise += cfg.noise_offset * torch.randn(
                            (noise.shape[0], noise.shape[1], 1, 1, 1),
                            device=noise.device,
                        )

                    bsz = latents.shape[0]
                    # Sample a random timestep for each video
                    timesteps = torch.randint(
                        0,
                        train_noise_scheduler.num_train_timesteps,
                        (bsz,),
                        device=latents.device,
                    )
                    timesteps = timesteps.long()

                    tgt_pose_img = batch["tgt_pose"]
                    tgt_pose_img = tgt_pose_img.unsqueeze(2)  # (bs, 3, 1, 512, 512)

                    uncond_fwd = random.random() < cfg.uncond_ratio
                    clip_image_list = []
                    ref_image_list = []
                    for batch_idx, (ref_img, clip_img) in enumerate(
                        zip(
                            batch["ref_img"],
                            batch["clip_images"],
                        )
                    ):
                        if uncond_fwd:
                            clip_image_list.append(torch.zeros_like(clip_img))
                        else:
                            clip_image_list.append(clip_img)
                        ref_image_list.append(ref_img)

                    with torch.no_grad():
                        ref_img = torch.stack(ref_image_list, dim=0).to(
                            dtype=vae.dtype, device=vae.device
                        )
                        ref_image_latents = vae.encode(
                            ref_img
                        ).latent_dist.sample()  # (bs, d, 64, 64)
                        ref_image_latents = ref_image_latents * 0.18215

                        clip_img = torch.stack(clip_image_list, dim=0).to(
                            dtype=image_enc.dtype, device=image_enc.device
                        )
                        clip_image_embeds = image_enc(
                            clip_img.to("cuda", dtype=weight_dtype)
                        ).image_embeds
                        image_prompt_embeds = clip_image_embeds.unsqueeze(1)  # (bs, 1, d)

                    # add noise
                    noisy_latents = train_noise_scheduler.add_noise(
                        latents, noise, timesteps
                    )

                    # Get the target for loss depending on the prediction type
                    if train_noise_scheduler.prediction_type == "epsilon":
                        target = noise
                    elif train_noise_scheduler.prediction_type == "v_prediction":
                        target = train_noise_scheduler.get_velocity(
                            latents, noise, timesteps
                        )
                    else:
                        raise ValueError(
                            f"Unknown prediction type {train_noise_scheduler.prediction_type}"
                        )
                    # print(noisy_latents.size(), noisy_latents.dtype, timesteps.size(), timesteps, \
                    #     ref_image_latents.size(),ref_image_latents.dtype, image_prompt_embeds.dtype,\
                    #     image_prompt_embeds.size(), tgt_pose_img.size(), tgt_pose_img.dtype, uncond_fwd)
                    model_pred = net(
                        noisy_latents,
                        timesteps,
                        ref_image_latents,
                        image_prompt_embeds,
                        tgt_pose_img,
                        uncond_fwd,
                    )

                    if cfg.snr_gamma == 0:
                        loss = F.mse_loss(
                            model_pred.float(), target.float(), reduction="mean"
                        )
                    else:
                        snr = compute_snr(train_noise_scheduler, timesteps)
                        if train_noise_scheduler.config.prediction_type == "v_prediction":
                            # Velocity objective requires that we add one to SNR values before we divide by them.
                            snr = snr + 1
                        mse_loss_weights = (
                            torch.stack(
                                [snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1
                            ).min(dim=1)[0]
                            / snr
                        )
                        loss = F.mse_loss(
                            model_pred.float(), target.float(), reduction="none"
                        )
                        loss = (
                            loss.mean(dim=list(range(1, len(loss.shape))))
                            * mse_loss_weights
                        )
                        loss = loss.mean()

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = accelerator.gather(loss.repeat(cfg.data.train_bs)).mean()
                    train_loss += avg_loss.item() / cfg.solver.gradient_accumulation_steps
                    if accelerator.is_main_process:
                        print('max mem after first FORWARD',torch.cuda.max_memory_allocated()/1e6)
                    # Backpropagate
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            trainable_params,
                            cfg.solver.max_grad_norm,
                        )
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    if accelerator.is_main_process:
                        print('max mem after first BACKWARD',torch.cuda.max_memory_allocated()/1e6)

                if accelerator.sync_gradients:
                    reference_control_reader.clear()
                    reference_control_writer.clear()
                    progress_bar.update(1)
                    global_step += 1
                    accelerator.log({"train_loss": train_loss}, step=global_step)
                    train_loss = 0.0
                    if global_step % cfg.checkpointing_steps == 0:
                        if accelerator.is_main_process:
                            save_path = os.path.join(save_dir, f"checkpoint-{global_step}")
                            delete_additional_ckpt(save_dir, 1)
                            accelerator.save_state(save_path)

                    if global_step % cfg.val.validation_steps == 0:
                        if accelerator.is_main_process:
                            generator = torch.Generator(device=accelerator.device)
                            generator.manual_seed(cfg.seed)
                            if cfg.data.validation_val_max_img>0:
                                sample_dicts, val_loss = log_validation(
                                    dataloader=validation_dataloader,
                                    vae=vae,
                                    image_enc=image_enc,
                                    net=net,
                                    scheduler=val_noise_scheduler,
                                    accelerator=accelerator,
                                    cfg=cfg,
                                    val_max_img=cfg.data.validation_val_max_img,
                                )

                                log_dir=Path(f"{save_dir}/{global_step:06d}/val")
                                os.makedirs(log_dir, exist_ok=True)
                                images={}
                                for sample_id, sample_dict in enumerate(sample_dicts):
                                    sample_name = sample_dict["name"]
                                    for img_name, img in sample_dict.items():
                                        if img_name=='name': continue
                                        if img_name not in images:
                                            images[img_name]=[]
                                        os.makedirs(os.path.join(log_dir, img_name), exist_ok=True)
                                        img = sample_dict[img_name]
                                        out_file = os.path.join(log_dir, img_name,str(sample_name)+'.gif')
                                        img.save(out_file)
                                        # mlflow.log_artifact(out_file)
                                        images[img_name].append(imageio.imread(out_file))
                                for video_name,v in images.items():
                                    imageio.mimsave(os.path.join(log_dir, video_name,'val.gif'), v)
                                if use_wandb:
                                    wandb.log({"valdata_val_loss":np.mean(val_loss)}, global_step)
                            
                            if cfg.data.train_val_max_img>0:
                                sample_dicts, val_loss = log_validation(
                                    dataloader=train_dataloader,
                                    vae=vae,
                                    image_enc=image_enc,
                                    net=net,
                                    scheduler=val_noise_scheduler,
                                    accelerator=accelerator,
                                    cfg=cfg,
                                    val_max_img=cfg.data.train_val_max_img,
                                )

                                log_dir=Path(f"{save_dir}/{global_step:06d}/train")
                                os.makedirs(log_dir, exist_ok=True)
                                images={}
                                for sample_id, sample_dict in enumerate(sample_dicts):
                                    sample_name = sample_dict["name"]
                                    for img_name, img in sample_dict.items():
                                        if img_name=='name': continue
                                        if img_name not in images:
                                            images[img_name]=[]
                                        os.makedirs(os.path.join(log_dir, img_name), exist_ok=True)
                                        img = sample_dict[img_name]
                                        out_file = os.path.join(log_dir, img_name,str(sample_name)+'.gif')
                                        img.save(out_file)
                                        # mlflow.log_artifact(out_file)
                                        images[img_name].append(imageio.imread(out_file))
                                for video_name,v in images.items():
                                    imageio.mimsave(os.path.join(log_dir, video_name,'train.gif'), v)
                                if use_wandb:
                                    wandb.log({"traindata_val_loss":np.mean(val_loss)}, global_step)

                            net.reference_control_writer=ReferenceAttentionControl(
                                reference_unet,
                                do_classifier_free_guidance=False,
                                mode="write",
                                fusion_blocks="full",
                            )
                            net.reference_control_reader = ReferenceAttentionControl(
                                denoising_unet,
                                do_classifier_free_guidance=False,
                                mode="read",
                                fusion_blocks="full",
                            )

                logs = {
                    "step_loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                progress_bar.set_postfix(**logs)

                if accelerator.is_main_process and use_wandb:
                    wandb.log({"train_loss": loss.detach().item()}, step=global_step)

                if global_step >= cfg.solver.max_train_steps:
                    break

            # save model after each epoch
            if (
                epoch + 1
            ) % cfg.save_model_epoch_interval == 0 and accelerator.is_main_process:
                unwrap_net = accelerator.unwrap_model(net)
                save_checkpoint(
                    unwrap_net.reference_unet,
                    save_dir,
                    "reference_unet",
                    global_step,
                    total_limit=3,
                )
                save_checkpoint(
                    unwrap_net.denoising_unet,
                    save_dir,
                    "denoising_unet",
                    global_step,
                    total_limit=3,
                )
                save_checkpoint(
                    unwrap_net.pose_guider,
                    save_dir,
                    "pose_guider",
                    global_step,
                    total_limit=3,
                )
    elif args.mode=='val':
        if accelerator.is_main_process:
            generator = torch.Generator(device=accelerator.device)
            generator.manual_seed(cfg.seed)
            if cfg.data.validation_val_max_img>0:
                sample_dicts, val_loss = log_validation(
                    dataloader=validation_dataloader,
                    vae=vae,
                    image_enc=image_enc,
                    net=net,
                    scheduler=val_noise_scheduler,
                    accelerator=accelerator,
                    cfg=cfg,
                    val_max_img=400,#cfg.data.validation_val_max_img,
                )

                log_dir=Path(f"{save_dir}/{global_step:06d}/val_test")
                log_time_dir=log_dir
                os.makedirs(log_dir, exist_ok=True)
                images={}
                for sample_id, sample_dict in enumerate(sample_dicts):
                    sample_name = sample_dict["name"]
                    for img_name, img in sample_dict.items():
                        if img_name=='name': continue
                        if img_name not in images:
                            images[img_name]=[]
                        os.makedirs(os.path.join(log_dir, img_name), exist_ok=True)
                        img = sample_dict[img_name]
                        out_file = os.path.join(log_dir, img_name,str(sample_name)+'.gif')
                        img.save(out_file)
                        # mlflow.log_artifact(out_file)
                        images[img_name].append(imageio.imread(out_file))
                for video_name,v in images.items():
                    imageio.mimsave(os.path.join(log_dir, video_name,'val.gif'), v)

                with open(os.path.join(log_dir,'valdata_val_loss.txt'), 'w') as f:
                    f.write(str(np.mean(val_loss)))
                    f.write(str(val_loss))
            
            if cfg.data.train_val_max_img>0:
                sample_dicts, val_loss = log_validation(
                    dataloader=train_dataloader,
                    vae=vae,
                    image_enc=image_enc,
                    net=net,
                    scheduler=val_noise_scheduler,
                    accelerator=accelerator,
                    cfg=cfg,
                    val_max_img=cfg.data.train_val_max_img,
                )

                log_dir=Path(f"{save_dir}/{global_step:06d}/train_test")
                os.makedirs(log_dir, exist_ok=True)
                images={}
                for sample_id, sample_dict in enumerate(sample_dicts):
                    sample_name = sample_dict["name"]
                    for img_name, img in sample_dict.items():
                        if img_name=='name': continue
                        if img_name not in images:
                            images[img_name]=[]
                        os.makedirs(os.path.join(log_dir, img_name), exist_ok=True)
                        img = sample_dict[img_name]
                        out_file = os.path.join(log_dir, img_name,str(sample_name)+'.gif')
                        img.save(out_file)
                        # mlflow.log_artifact(out_file)
                        images[img_name].append(imageio.imread(out_file))
                for video_name,v in images.items():
                    imageio.mimsave(os.path.join(log_dir, video_name,'train.gif'), v)
                
                with open(os.path.join(log_dir,'traindata_val_loss.txt'), 'w') as f:
                    f.write(str(np.mean(val_loss)))
                    f.write(str(val_loss))
            print('max mem after Val',torch.cuda.max_memory_allocated()/1e6)

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()
    return log_time_dir


def save_checkpoint(model, save_dir, prefix, ckpt_num, total_limit=None):
    save_path = osp.join(save_dir, f"{prefix}-{ckpt_num}.pth")

    if total_limit is not None:
        checkpoints = os.listdir(save_dir)
        checkpoints = [d for d in checkpoints if d.startswith(prefix)]
        checkpoints = sorted(
            checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0])
        )

        if len(checkpoints) >= total_limit:
            num_to_remove = len(checkpoints) - total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(save_dir, removing_checkpoint)
                os.remove(removing_checkpoint)

    state_dict = model.state_dict()
    torch.save(state_dict, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/training/stage1.yaml")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--train_val_max_img", type=int, default=0)
    parser.add_argument("--validation_val_max_img", type=int, default=500)
    parser.add_argument("--validation_steps", type=int, default=1)
    parser.add_argument("--resume_ckpt", type=str, default="latest")
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--use_wandb",action='store_true')
    parser.add_argument("--res_512",action='store_true')
    parser.add_argument("--AA",action='store_true')
    args = parser.parse_args()

    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
    elif args.config[-3:] == ".py":
        config = import_filename(args.config).cfg
    else:
        raise ValueError("Do not support this format config file")

    if args.res_512:
        config.data.train_width=512
        config.data.train_height=512
        config.train_data.params.sample_size=512
        config.validation_data.params.sample_size=512
        # if 'sample_finetune' in args.config:
        if 'train' in args.config:
            config.train_data.params.H=512
            config.train_data.params.W=512
            config.train_data.params.cond_H=64
            config.train_data.params.cond_W=64
            # actually params.sample_size is not used in train configs
            config.validation_data.params.H=512
            config.validation_data.params.W=512
            config.validation_data.params.cond_H=64
            config.validation_data.params.cond_W=64
    if args.AA:
        assert config.train_data.target=='src.dataset.ChDataset_GaussianSkeleton_DiverseAppearance_json_rndLayerOrder.ChDataset'
        assert args.res_512==True, 'You are finetuning animate anyone with resonlution not 512! Make sure you want to do this.'
        config.base_model_path= "./pretrained_weights/stable-diffusion-v1-5/"
        config.vae_model_path= "./pretrained_weights/sd-vae-ft-mse"
        config.image_encoder_path= "./pretrained_weights/image_encoder"
        config.denoising_unet_path= "./pretrained_weights/denoising_unet.pth"
        config.reference_unet_path= "./pretrained_weights/reference_unet.pth"
        config.pose_guider_path= "./pretrained_weights/pose_guider.pth"
        config.controlnet_openpose_path='./pretrained_weights/control_v11p_sd15_openpose/diffusion_pytorch_model.bin'
        config.pose_guider_pretrain=True
        config.data.train_bs=1
        config.data.validation_val_max_img=2
        config.data.train_val_max_img=4

        config.resume_from_checkpoint=''
        config.save_name='stage1'
        config.output_dir='./results/finetune_AA/'+args.config.split('.yaml')[0].split('_')[-1]

        config.train_data.params.draw_joint_skeleton='openpose'
        # config.train_data.params.transform_norm=True
        config.validation_data.params.draw_joint_skeleton='openpose'
        # config.validation_data.params.transform_norm=True

        config.solver.max_train_steps=2000

    if args.mode=='val':
        config.data.train_val_max_img=args.train_val_max_img
        config.data.validation_val_max_img=args.validation_val_max_img
        config.val.validation_steps=args.validation_steps
        config.resume_from_checkpoint=args.resume_ckpt
        config.validation_data.params.rotate_translate_prob=0.
        config.exp_name=config.save_name

    start=time.time()
    log_time_dir=main(args, config)
    duration=time.time()-start
    time_now = datetime.now()
    formatted_datetime = str(time_now.strftime("%Y-%m-%d_%H:%M:%S"))
    with open(os.path.join(log_time_dir,'runtime_'+formatted_datetime+'.txt'),'w') as f:
        f.write(str(duration))