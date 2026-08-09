import argparse
import copy
from copy import deepcopy
import logging
import os
import shutil
from pathlib import Path
from collections import OrderedDict
import json

import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from models.sit import SiT_models
from loss import SILoss, mean_flat

from data.mmgen_dataset import MMGenDataset
from diffusers.models import AutoencoderKL
from torchvision.utils import save_image

from utils.checkpoints import load_checkpoint
from utils.latents import sample_posterior
import data.path as PathConfig

logger = get_logger(__name__)

SNAPSHOT_PATHS = ("models", "data", "loss.py", "samplers.py", "train.py", "utils")


def validate_training_args(args):
    if args.resume_step < 0:
        raise ValueError("--resume-step must be non-negative")
    if args.resume_step and args.ckpt:
        raise ValueError("Use either --resume-step or --ckpt, not both")
    if args.num_tasks not in range(1, len(PathConfig.TASK_NAMES) + 1):
        raise ValueError(f"--num-tasks must be between 1 and {len(PathConfig.TASK_NAMES)}")
    if args.is_mmcat_diff != (args.num_tasks > 1):
        raise ValueError(
            "Use --is-mmcat-diff exactly when --num-tasks is greater than 1"
        )
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    for name in ("epochs", "max_train_steps", "checkpointing_steps", "sampling_steps"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not 0.0 <= args.cfg_prob <= 1.0:
        raise ValueError("--cfg-prob must be between 0 and 1")


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        if not param.requires_grad:
            continue
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def snapshot_sources(destination):
    """Save the source files needed to reproduce a training run."""
    os.makedirs(destination, exist_ok=True)
    for source in SNAPSHOT_PATHS:
        target = os.path.join(destination, source)
        if os.path.isdir(source):
            shutil.copytree(
                source,
                target,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(
                    "local", "__pycache__", "*.h5", "*.pt", "*.pth"
                ),
            )
        else:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copy2(source, target)


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):    
    validate_training_args(args)
    # set accelerator
    args.logging_dir = args.exp_name  + '/' + args.logging_dir 
    
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
        
        sample_dir = f"{save_dir}/samples"
        os.makedirs(sample_dir, exist_ok=True)
        logger.info(f"Sample directory created at {sample_dir}")
        
        
        snapshot_sources(os.path.join(save_dir, "code"))
        
        
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False    
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)
    
    # Create model:
    assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.resolution // 8
    
    
    # args
    num_tasks = args.num_tasks
    is_mmcat_diff = args.is_mmcat_diff
    use_decouple_task = args.use_decouple_task
    ratio_train_onlyrgb = args.ratio_train_onlyrgb
    use_drop_input = args.use_drop_input
    
    use_decouple_rand1 = args.use_decouple_rand1
    use_taskcond_emb = args.use_taskcond_emb
    
    use_decouple_latent = args.use_decouple_latent
    mode_time_emb = args.mode_time_emb
    use_vel_comp = args.use_vel_comp
    use_diff_task_weight = args.use_diff_task_weight
    use_half_projloss = args.use_half_projloss
    use_quater_mixing = args.use_quater_mixing

    lis_weight = [1.0, 1.0, 1.0, 1.0]
    if use_diff_task_weight:
        lis_weight = [1.0, 0.2, 0.1, 0.1]
    print(f'\n\n******** list of task weight: {lis_weight}. **************\n\n')
                        
    z_dims = [768]
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        class_dropout_prob=args.cfg_prob,
        z_dims = z_dims,
        encoder_depth=args.encoder_depth,
        num_tasks=num_tasks,
        is_mmcat_diff=is_mmcat_diff,
        use_decouple_task=use_decouple_task,
        use_decouple_latent=use_decouple_latent,
        mode_time_emb=mode_time_emb,
        use_taskcond_emb=use_taskcond_emb,
        **block_kwargs
    )

    model = model.to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training

    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)
    requires_grad(ema, False)
    
    latents_scale = torch.tensor(
        [0.18215, 0.18215, 0.18215, 0.18215]
        ).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor(
        [0., 0., 0., 0.]
        ).view(1, 4, 1, 1).to(device)

    # create loss function
    loss_fn = SILoss(
        prediction=args.prediction,
        path_type=args.path_type, 
        weighting=args.weighting,
        use_decouple_task=use_decouple_task,
        num_tasks=num_tasks,
        ratio_train_onlyrgb=ratio_train_onlyrgb,
        use_drop_input=use_drop_input,
        use_decouple_latent=use_decouple_latent,
        mode_time_emb=mode_time_emb,
        use_vel_comp=use_vel_comp,
        use_half_projloss=use_half_projloss,
        use_quater_mixing=use_quater_mixing,
        use_decouple_rand1=use_decouple_rand1,
        cond_task_fixed=args.cond_task_fixed,
    )
    if accelerator.is_main_process:
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )    
    
    # Setup data:
    vae_h5_paths = {
        'rgb': args.vae_rgb_h5,
        'depth': args.vae_depth_h5,
        'normal': args.vae_normal_h5,
        'mask': args.vae_mask_h5,
    }
    train_dataset = MMGenDataset(
                            path_meta_data=args.metadata_path,
                            num_tasks=num_tasks, 
                            use_preprocess_vae=True,
                            use_repa_reg=True,
                            vae_h5_paths=vae_h5_paths,
                            dino_rgb_h5=args.dino_rgb_h5)
    if args.batch_size % accelerator.num_processes:
        raise ValueError("--batch-size must be divisible by the number of processes")
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    print(f'Local batch size: {local_batch_size} with learning rate: {args.learning_rate}')

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} samples ({args.metadata_path})")
    
    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    
    # resume:
    global_step = 0; start_step = 0
    if args.resume_step > 0 or args.ckpt is not None:
        if args.resume_step > 0:
            ckpt_name = str(args.resume_step).zfill(7) +'.pt'
            ckpt = load_checkpoint(
                f'{os.path.join(args.output_dir, args.exp_name)}/checkpoints/{ckpt_name}'
            )
            
            model.load_state_dict(ckpt['model'])

        if args.ckpt is not None:
            ckpt = load_checkpoint(args.ckpt)
            print(f'Load ckpt from {args.ckpt}')

        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['opt'])
        # loaded opt state carries the saved lr; override with the requested one
        for pg in optimizer.param_groups:
            pg['lr'] = args.learning_rate
        print(f'Override optimizer lr -> {args.learning_rate} after ckpt load')

        global_step = ckpt['steps']
        start_step = global_step
        print(f'Continue train from step {global_step}')

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(
            project_name="REPA", 
            config=tracker_config,
        )
        print(f'Rank: {accelerator.process_index}')
 
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        initial=global_step,
        total=args.max_train_steps,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    sample_batch_size = 8
    dict0 = next(iter(train_dataloader))
    gt_xs = dict0['x']
    gt_xs = gt_xs[:sample_batch_size]
    gt_xs = sample_posterior(
        gt_xs.to(device), latents_scale=latents_scale, latents_bias=latents_bias,
        is_mmcat_diff=is_mmcat_diff, num_tasks=num_tasks
        )    
    for epoch in range(args.epochs):
        model.train()
        for dict_data_inp in train_dataloader:
            x, y = dict_data_inp['x'], dict_data_inp['idx_cat']
            with torch.no_grad():
                x = sample_posterior(x, latents_scale=latents_scale, latents_bias=latents_bias, 
                                        is_mmcat_diff=is_mmcat_diff, num_tasks=num_tasks)
                
            z = dict_data_inp['x_repa'].to(device)
            zs = [z]
            labels = y
            
            with accelerator.accumulate(model):
                model_kwargs = dict(y=labels)
                loss, proj_loss, mask_drop = loss_fn(model, x, model_kwargs, zs=zs)
                
                dict_log_mm = {}
                if is_mmcat_diff:
                    # Log detached loss values for each task.
                    lis_loss = loss.chunk(num_tasks, dim=1)
                    if mask_drop is not None:
                        lis_mask_drop = mask_drop[...,0,0].bool().chunk(num_tasks, dim=1)
            
                    loss_mean = 0            
                    for i in range(num_tasks):
                        loss_i = mean_flat(lis_loss[i])
                        if mask_drop is not None:
                            loss_i  = loss_i[lis_mask_drop[i][...,0]]
                            n_elem_rest = loss_i.numel()
                            if n_elem_rest==0:
                                if i==0:
                                    dict_log_mm.update({"loss": 0.0})
                                continue  # continue if all elements are dropped
                        
                        weight_i = lis_weight[i]
                        loss_i = loss_i.mean()
                        loss_mean += weight_i*loss_i.mean()
                        
                        dict_log_mm.update({f"loss/task_{i}": loss_i.item()})
                        if i==0:
                            dict_log_mm.update({"loss": loss_i.item()})
                    dict_log_mm.update({"loss/task_all": loss_mean.item()})

                    loss = loss_mean
                    # loss = mean_flat(loss).mean()
                else:
                    loss = mean_flat(loss).mean()
                    dict_log_mm.update({"loss/task_0": loss.item()})
                    
                        
                
                loss_mean = loss.mean()
                proj_loss_mean = proj_loss.mean()
                loss = loss_mean + proj_loss_mean * args.proj_coeff
                    
                ## optimization
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    update_ema(ema, model) # change ema function

            # Checkpointing, previews, and logging are defined on optimizer steps.
            if not accelerator.sync_gradients:
                continue

            progress_bar.update(1)
            global_step += 1
            if global_step % args.checkpointing_steps == 0 and global_step > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": accelerator.unwrap_model(model).state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": vars(copy.deepcopy(args)),
                        "steps": global_step,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            if (global_step == (start_step+1) or (global_step % args.sampling_steps == 0 and global_step > 0)):
                from samplers import euler_sampler
                if accelerator.is_main_process:
                    print("Generating EMA samples...")
                
                for i_infer in tqdm(range(1)):

                    ys = torch.randint(args.num_classes, size=(sample_batch_size,), device=device)
                    # Create sampling noise:
                    n = ys.size(0)
                    xT = torch.randn((n, 4*num_tasks, latent_size, latent_size), device=device)

                    with torch.no_grad():
                        samples = euler_sampler(
                            ema,
                            xT, 
                            ys,
                            num_steps=50, 
                            cfg_scale=4.0 if args.cfg_prob > 0 else 1.0,
                            guidance_low=0.,
                            guidance_high=1.,
                            path_type=args.path_type,
                            heun=False,
                            use_decouple_task=use_decouple_task,
                            use_decouple_latent=use_decouple_latent,
                        ).to(torch.float32)
                        if is_mmcat_diff:
                            lis_samples = samples.chunk(num_tasks, dim=1)
                            samples = torch.cat(lis_samples, dim=0)

                            gt_latents = x[:sample_batch_size] if global_step > 100 else gt_xs
                            gt_latents = torch.cat(gt_latents.chunk(num_tasks, dim=1), dim=0)
                        else:
                            gt_latents = gt_xs
                            
                        samples = vae.decode((samples -  latents_bias) / latents_scale).sample
                        gt_samples = vae.decode((gt_latents - latents_bias) / latents_scale).sample
                        samples = (samples + 1) / 2.
                        gt_samples = (gt_samples + 1) / 2.
                    out_samples = accelerator.gather(samples.to(torch.float32))
                    gt_samples = accelerator.gather(gt_samples.to(torch.float32))
                    if accelerator.is_main_process:
                        save_image(out_samples, f"{sample_dir}/sample_{global_step:07d}_iter{i_infer:04d}.png", nrow=8, normalize=False, value_range=(0, 1))
                        # if global_step <=1:
                        save_image(gt_samples, f"{sample_dir}/gt_sample_{global_step:07d}_iter{i_infer:04d}.png", nrow=8, normalize=False, value_range=(0, 1))
                logging.info("Generating EMA samples done.")

            logs = {
                "loss/loss_all": accelerator.gather(loss_mean).mean().detach().item(), 
                "proj_loss": accelerator.gather(proj_loss_mean).mean().detach().item(),
                "grad_norm": accelerator.gather(grad_norm).mean().detach().item()
            }
            logs.update(dict_log_mm)
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        
        # log epoch
        accelerator.log({"log/epoch": epoch}, step=global_step)
            
        if global_step >= args.max_train_steps:
            break

    model.eval()  # important! This disables randomized embedding dropout
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging:
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="tensorboard")
    parser.add_argument("--sampling-steps", type=int, default=10000)
    parser.add_argument("--resume-step", type=int, default=0)

    # model
    parser.add_argument("--model", choices=sorted(SiT_models), required=True)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm",  action='store_true', default=False)

    # dataset
    parser.add_argument("--metadata-path", type=str, default=PathConfig.path_meta_data)
    parser.add_argument("--vae-rgb-h5", type=str, default=PathConfig.path_h5_rgb)
    parser.add_argument("--vae-depth-h5", type=str, default=PathConfig.path_h5_depth)
    parser.add_argument("--vae-normal-h5", type=str, default=PathConfig.path_h5_normal)
    parser.add_argument("--vae-mask-h5", type=str, default=PathConfig.path_h5_mask)
    parser.add_argument("--dino-rgb-h5", type=str, default=PathConfig.path_h5_dino_rgb)
    parser.add_argument(
        "--vae-path", type=str, default="stabilityai/sd-vae-ft-mse",
        help="Local directory or Hugging Face model id used for sample previews.",
    )
    parser.add_argument("--resolution", type=int, choices=[256], default=256)
    parser.add_argument("--batch-size", type=int, default=256)

    # precision
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # optimization
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=400000)
    parser.add_argument("--checkpointing-steps", type=int, default=50000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")

    # seed
    parser.add_argument("--seed", type=int, default=0)

    # cpu
    parser.add_argument("--num-workers", type=int, default=4)

    # loss
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"]) # currently we only support v-prediction
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--proj-coeff", type=float, default=0.5)
    parser.add_argument("--weighting", default="uniform", choices=["uniform"])
    # num_tasks, is-mmcat-diff
    parser.add_argument("--num-tasks", type=int, default=1)
    parser.add_argument("--is-mmcat-diff", action='store_true', default=False)
    
    # use_decouple_task
    parser.add_argument("--use-decouple-task", action='store_true', default=False)
    # use half projloss
    parser.add_argument("--use-half-projloss", action='store_true', default=False)
    # use quater mixing
    parser.add_argument(
        "--use-quarter-mixing",
        "--use-quater-mixing",
        dest="use_quater_mixing",
        action="store_true",
        default=False,
    )
    # use_decouple_rand1
    parser.add_argument("--use-decouple-rand1", action='store_true', default=False)
    # use_taskcond_emb
    parser.add_argument("--use-taskcond-emb", action='store_true', default=False)
    
    # use decouple v2
    parser.add_argument("--use-decouple-latent", action='store_true', default=False)
    # mode time emb, None
    parser.add_argument("--mode-time-emb", type=str, default='None', help='Mode time emb')
    # use_vel_comp
    parser.add_argument("--use-vel-comp", action='store_true', default=False)
    
    # use_diff_task_weight
    parser.add_argument("--use-diff-task-weight", action='store_true', 
                            default=False, help='Use diff task weight [1.0, 0.2, 0.1, 0.1]')
    
    # use_half_origsit
    parser.add_argument("--ratio-train-onlyrgb", type=float, default=0.0, help='Ratio train only rgb')
    # rgb->X (A): fix the condition task (e.g. 0=rgb); supervise only the others. -1 disables.
    parser.add_argument("--cond-task-fixed", type=int, default=-1,
                            help='Fix condition task idx for rgb->X training (0=rgb). -1 = disabled. Requires --use-decouple-rand1.')
    # use-drop-input
    parser.add_argument("--use-drop-input", action='store_true', default=False)

    ## ckpt
    parser.add_argument("--ckpt", type=str, default=None)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
        
    return args

if __name__ == "__main__":
    args = parse_args()
    
    main(args)
