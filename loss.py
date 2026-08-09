import torch
import numpy as np

import torch as th


TIME_EMBED_MODES = {
    "INSYNC_NONE",
    "INSYNC_VEC",
    "INSYNC_PATCH",
    "INSYNC_ALL",
    "INSYNC_TASK",
}

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

class SILoss:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            use_decouple_task=False,
            num_tasks=1,
            ratio_train_onlyrgb=0.0,
            use_drop_input=False,
            use_decouple_latent=False,
            mode_time_emb=None,
            use_vel_comp=False,
            use_half_projloss=False,
            use_quater_mixing=False,
            use_decouple_rand1=False,
            cond_task_fixed=-1,
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.use_decouple_task = use_decouple_task
        self.num_tasks = num_tasks
        self.use_quater_mixing = use_quater_mixing
        print('Use quarter mixing: ', self.use_quater_mixing)
        
        # augmentation
        self.ratio_train_onlyrgb = ratio_train_onlyrgb
        self.use_drop_input = use_drop_input
        print('Use drop input: ', self.use_drop_input)
        
        self.mode_time_emb = mode_time_emb
        self.use_decouple_latent = use_decouple_latent
        
        self.use_decouple_rand1 = use_decouple_rand1
        if self.use_decouple_rand1 and not self.use_decouple_task:
            raise ValueError("use_decouple_rand1 requires use_decouple_task")
        if self.use_decouple_rand1:
            print('Use decouple rand1: ', self.use_decouple_rand1)

        # Fixed-condition mode for rgb->X (A): always treat one task (idx) as the
        # clean generation condition, supervise only the other tasks. -1 = disabled.
        self.cond_task_fixed = cond_task_fixed
        if not -1 <= self.cond_task_fixed < self.num_tasks:
            raise ValueError(
                f"cond_task_fixed must be -1 or a task index below {self.num_tasks}"
            )
        if self.cond_task_fixed >= 0 and not self.use_decouple_rand1:
            raise ValueError("cond_task_fixed requires use_decouple_rand1")
        if self.cond_task_fixed >= 0:
            print(f'Fixed condition task: {self.cond_task_fixed} (rgb->X mode, no mixing)')

        if not 0.0 <= self.ratio_train_onlyrgb <= 1.0:
            raise ValueError("ratio_train_onlyrgb must be between 0 and 1")
        if self.use_decouple_latent and self.mode_time_emb not in TIME_EMBED_MODES:
            raise ValueError(
                f"mode_time_emb must be one of {sorted(TIME_EMBED_MODES)} when use_decouple_latent is enabled"
            )
        
        self.use_halfmixing = True
        self.use_blend = False
        self.use_vel_comp = use_vel_comp
        
        self.use_half_projloss = use_half_projloss

    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t


    def sample(self, x1):
        """Sampling x0 & t based on shape of x1 (if needed)
          Args:
            x1 - data point; [batch, *dim]
        """
        
        x0 = th.randn_like(x1)  # N*C*H*W
        # t0, t1 = self.check_interval(self.train_eps, self.sample_eps)
        t0 = 0.0; t1 = 1.0
        t_ori = th.rand(x1.shape[0], 1, 1, 1) * (t1 - t0) + t0
        t_ori = t_ori.repeat(1, x1.shape[1], x1.shape[2], x1.shape[3])

        if self.mode_time_emb == 'INSYNC_ALL':
            t = th.rand(x1.shape) * (t1 - t0) + t0
            
        elif self.mode_time_emb == 'INSYNC_VEC':
            t = th.rand(x1.shape[0], x1.shape[1], 1, 1) * (t1 - t0) + t0
            t = t.repeat(1, 1, x1.shape[2], x1.shape[3])
            
        elif self.mode_time_emb == 'INSYNC_PATCH':
            t = th.rand(x1.shape[0], 1, x1.shape[2], x1.shape[3]) * (t1 - t0) + t0
            t = t.repeat(1, x1.shape[1], 1, 1)
            
        elif self.mode_time_emb == 'INSYNC_NONE':
            t = th.rand(x1.shape[0], 1, 1, 1) * (t1 - t0) + t0
            t = t.repeat(1, x1.shape[1], x1.shape[2], x1.shape[3])
  
        elif self.mode_time_emb == 'INSYNC_TASK':
            t = th.rand(x1.shape[0], self.num_tasks, 1,  1, 1) * (t1 - t0) + t0  # N*M*C*H*W
            
            C_dim = x1.shape[1] // self.num_tasks
            assert x1.shape[1] % self.num_tasks == 0, 'C_dim should be divisible by num_tasks'
            t = t.repeat(1, 1, C_dim, x1.shape[2], x1.shape[3])
            
            t = t.chunk(self.num_tasks, dim=1)
            t = torch.cat(t, dim=2).squeeze(1)             
        else:
            raise NotImplementedError
        
        if self.use_halfmixing and self.mode_time_emb != "INSYNC_NONE":
            half = x1.shape[0]//2
            t = th.concat((t_ori[:half,...], t[half:,...]), dim=0)

        
        if self.use_blend and self.mode_time_emb != "INSYNC_NONE":
            # t blend scheme
            progress = th.rand(1)
            t = t_ori * (1-progress) + t * progress   # alleviate the blending effect
            # half-half scheme
            # half = x1.shape[0]//2
            # t = th.concat((t[:half,...], t_ori[half:,...]), dim=0)

        t = t.to(x1)
        return t, x0, x1

    def compensate_offdiagonal_ut(self, x1, xt, ut):
        """Compensate the off-diagonal velocity with to-diagonal residual
        x1: noise
        xt: blend
        ut: velocity, noise-image
        """
        delta = x1 - xt
        compensation = delta - \
                (th.einsum("nchw,nchw->n", delta, ut) / th.einsum("nchw,nchw->n", ut, ut)).view(-1,1,1,1) * ut
        ut_cmpn = ut + compensation
        half = x1.shape[0]//2
        ut = th.concat((ut[:half,...], ut_cmpn[half:,...]), dim=0)
        return ut

    def __call__(self, model, images, model_kwargs=None, zs=None):
        if model_kwargs is None:
            model_kwargs = {}
        # sample timesteps
        if self.weighting == "uniform":
            time_input = torch.rand(
                (images.shape[0], 1, 1, 1), device=images.device, dtype=images.dtype
            )
            
            if self.use_decouple_task:
                time_input_ori = torch.rand(
                    (images.shape[0], 1, 1, 1), device=images.device, dtype=images.dtype
                ).repeat(1, images.shape[1], 1, 1)
                
                C_dim = images.shape[1] // self.num_tasks
                assert C_dim == 4, "Only support 4 channels for now"
                
                time_input = torch.rand(
                    (images.shape[0], self.num_tasks, 1, 1, 1),
                    device=images.device,
                    dtype=images.dtype,
                )
                time_input = time_input.repeat(1, 1, C_dim, 1, 1)
                
                time_input = time_input.view(images.shape[0], -1, 1, 1)
                
                use_decouple_rand1 = self.use_decouple_rand1; mask_insync = None; lis_idx_taskcond_rand = None
                if use_decouple_rand1:
                    # only randomize one task, other three keep the same, this is similar for test
                    N_batch = images.shape[0]
                    if self.cond_task_fixed >= 0:
                        # rgb->X (A): every sample conditions on the same fixed task
                        lis_idx_taskcond_rand = torch.full(
                            (N_batch,), self.cond_task_fixed,
                            dtype=torch.long, device=images.device,
                        )
                    else:
                        lis_idx_taskcond_rand = torch.randint(
                            0, self.num_tasks, (N_batch,), device=images.device
                        )
                    
                    mask_insync = torch.zeros(
                        (N_batch, self.num_tasks), device=images.device, dtype=images.dtype
                    )
                    mask_insync[torch.arange(N_batch, device=images.device), lis_idx_taskcond_rand] = 1.0
                    
                    mask_insync = mask_insync.view(N_batch, self.num_tasks, 1, 1, 1)
                    mask_insync = mask_insync.repeat(1, 1, C_dim, 1, 1)
                    mask_insync = mask_insync.view(N_batch, -1, 1, 1)
                    
                    #
                    # time_input_rand1 = torch.rand((N_batch, 1, 1, 1))*time_input_ori[:,:1]
                    if self.cond_task_fixed >= 0:
                        # rgb->X (A): condition task pinned to t=0 (fully clean)
                        time_input_rand1 = torch.zeros(
                            (N_batch, 1, 1, 1), device=images.device, dtype=images.dtype
                        )
                    else:
                        time_input_rand1 = torch.rand(
                            (N_batch, 1, 1, 1), device=images.device, dtype=images.dtype
                        ) * 0.01
                    time_input_rand1 = time_input_rand1.repeat(1, images.shape[1], 1, 1)
                    
                    time_input = time_input_ori.clone()
                    time_input[mask_insync==1] = time_input_rand1[mask_insync==1]
                    
                # print(time_input[0].reshape(-1))
                # print(time_input[1].reshape(-1))
                                
                use_halfmixing = True
                if self.cond_task_fixed >= 0:
                    # rgb->X (A): no mixing — keep all samples in fixed-condition mode
                    pass
                elif self.use_quater_mixing:
                    half = images.shape[0]//2
                    quarter = images.shape[0]//4
                    time_input = torch.cat((time_input_ori[:(quarter+half)], time_input[(half+quarter):]), dim=0)
                    
                    if self.use_decouple_rand1:
                        mask_insync[:(quarter+half)] = 0.0
                        lis_idx_taskcond_rand[:(quarter+half)] = -1
                
                elif use_halfmixing:
                    half = images.shape[0]//2
                    time_input = torch.cat((time_input_ori[:half], time_input[half:]), dim=0)
                    
                    if self.use_decouple_rand1:
                        mask_insync[:(half)] = 0.0
                        lis_idx_taskcond_rand[:(half)] = -1
                        
                use_blend = False
                if use_blend:
                    progress = torch.rand(1, device=images.device, dtype=images.dtype)
                    time_input = time_input_ori * (1-progress) + time_input * progress
                
                if self.use_decouple_rand1:
                    model_kwargs['lis_idx_taskcond_rand'] = lis_idx_taskcond_rand + 1
            
            if self.use_decouple_latent:
                time_input = self.sample(images)[0]
                
        elif self.weighting == "lognormal":
            raise NotImplementedError("lognormal weighting is not implemented")
        else:
            raise ValueError(f"Unknown weighting: {self.weighting}")
                
        time_input = time_input.to(device=images.device, dtype=images.dtype)
        
        noises = torch.randn_like(images)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)
            
        model_input = alpha_t * images + sigma_t * noises
        
        n_train_onlyrgb = 0; mask_valid = None; is_drop_input = self.use_drop_input
        if self.cond_task_fixed >= 0:
            # rgb->X (A): supervise only the non-condition tasks. mask_insync==1 marks
            # the fixed condition task (rgb), whose velocity loss is zeroed out.
            mask_valid = torch.ones_like(model_input)
            mask_insync_flat = mask_insync.squeeze(-1).squeeze(-1)
            mask_valid[mask_insync_flat == 1] = 0.0
        if self.ratio_train_onlyrgb > 0.0:
            n_train_onlyrgb = int(images.shape[0] * self.ratio_train_onlyrgb)

            mask_valid = torch.ones_like(model_input)
            if self.use_decouple_rand1:
                # no velocity supervision for the insync task, in other words, this task is regarded as a generation condition
                mask_insync = mask_insync.squeeze(-1).squeeze(-1)
                mask_valid[mask_insync==1] = 0.0
            
            mask_valid[:n_train_onlyrgb, 4:] = 0.0

            # drop input
            if is_drop_input:
                model_input[:n_train_onlyrgb, 4:] = 0.0
   
        if (
            0 < self.ratio_train_onlyrgb < 0.9
            and int(images.shape[0] * (1 - self.ratio_train_onlyrgb)) > 0
        ):
            # radomly select a task to drop
            ratio_randrop_task = 1 - self.ratio_train_onlyrgb
            N1 = images.shape[0]
            n_use_randrop = int(N1 * ratio_randrop_task)
            
            N_start_randrop = n_train_onlyrgb
            N_end_randrop = n_train_onlyrgb + n_use_randrop
            
            # get random drop index
            n_task_all = n_use_randrop*self.num_tasks
            n_drop_rand = np.random.choice(n_task_all, 1, replace=False)[0]   
            n_drop_rand = min(n_drop_rand, n_task_all-1)
            if not self.use_decouple_rand1:
                n_drop_rand = max(n_drop_rand, n_use_randrop)
            idx_drop_rand = np.random.choice(n_task_all, n_drop_rand, replace=False)
            idx_drop_rand = np.setdiff1d(idx_drop_rand, np.arange(0, n_task_all, self.num_tasks))
            
            # generate random drop mask
            mask_randdrop = torch.ones((n_task_all,)).to(device=images.device)
            mask_randdrop[idx_drop_rand] = 0.0
            
            mask_randdrop = mask_randdrop.view(n_use_randrop, self.num_tasks, 1, 1, 1)
            mask_randdrop = mask_randdrop.repeat(1, 1, 4, 1, 1)
            mask_randdrop = mask_randdrop.view(n_use_randrop, -1, 1, 1)
            
            # drop input
            if is_drop_input:
                model_input[N_start_randrop:N_end_randrop] = model_input.clone()[N_start_randrop:N_end_randrop] * mask_randdrop

            # update merged mask_valid
            mask_valid[N_start_randrop:N_end_randrop] = mask_valid.clone()[N_start_randrop:N_end_randrop] * mask_randdrop
      
        if self.prediction == 'v':
            model_target = d_alpha_t * images + d_sigma_t * noises
            if self.use_vel_comp:
                model_target = self.compensate_offdiagonal_ut(noises, model_input, model_target)
        else:
            raise NotImplementedError(f"Prediction mode is not implemented: {self.prediction}")
        
        if not (self.use_decouple_task or self.use_decouple_latent):
            time_input = time_input.flatten()
            
        model_output, zs_tilde  = model(model_input, time_input, **model_kwargs)
        # denoising_loss = mean_flat((model_output - model_target) ** 2)
        denoising_loss = (model_output - model_target) ** 2
        
        # drop noise
        if self.ratio_train_onlyrgb > 0.0 or self.cond_task_fixed >= 0:
            denoising_loss = denoising_loss * mask_valid

        # projection loss
        proj_loss = 0.
        bsz = zs[0].shape[0]
        samples_used = bsz // 2 if self.use_half_projloss else bsz
        if samples_used == 0:
            raise ValueError("use_half_projloss requires a batch size of at least 2")
        for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):
            for z_j, z_tilde_j in zip(z[:samples_used], z_tilde[:samples_used]):
                z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1) 
                z_j = torch.nn.functional.normalize(z_j, dim=-1) 
                proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))

        proj_loss /= (len(zs) * samples_used)

        return denoising_loss, proj_loss, mask_valid
