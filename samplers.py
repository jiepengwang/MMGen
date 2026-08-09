"""ODE and Euler-Maruyama samplers for SiT models."""

import torch


def expand_t_like_x(t, x):
    """Reshape a batch timestep vector so it broadcasts over `x`."""
    if t.ndim == x.ndim:
        return t
    return t.view(t.shape[0], *([1] * (x.ndim - 1)))


def get_score_from_velocity(velocity, sample, timestep, path_type="linear"):
    """Convert a velocity prediction to a score for the selected path."""
    timestep = expand_t_like_x(timestep, sample)
    if path_type == "linear":
        alpha = 1 - timestep
        d_alpha = -torch.ones_like(sample)
        sigma = timestep
        d_sigma = torch.ones_like(sample)
    elif path_type == "cosine":
        alpha = torch.cos(timestep * torch.pi / 2)
        sigma = torch.sin(timestep * torch.pi / 2)
        d_alpha = -torch.pi / 2 * torch.sin(timestep * torch.pi / 2)
        d_sigma = torch.pi / 2 * torch.cos(timestep * torch.pi / 2)
    else:
        raise ValueError(f"Unknown path type: {path_type}")

    reverse_alpha_ratio = alpha / d_alpha
    variance = sigma.square() - reverse_alpha_ratio * d_sigma * sigma
    return (reverse_alpha_ratio * velocity - sample) / variance


def _model_timestep(model_input, timestep, use_decouple_task, use_decouple_latent):
    batch_size = model_input.shape[0]
    kwargs = {
        "device": model_input.device,
        "dtype": model_input.dtype,
    }
    if use_decouple_latent:
        return torch.ones_like(model_input) * timestep
    if use_decouple_task:
        return torch.ones((batch_size, model_input.shape[1], 1, 1), **kwargs) * timestep
    return torch.ones((batch_size,), **kwargs) * timestep


@torch.no_grad()
def euler_sampler(
    model,
    latents,
    y,
    num_steps=20,
    heun=False,
    cfg_scale=1.0,
    guidance_low=0.0,
    guidance_high=1.0,
    path_type="linear",
    use_decouple_task=False,
    use_decouple_latent=False,
):
    """Integrate the SiT velocity field from noise to data with Euler/Heun."""
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1")
    if cfg_scale < 1.0:
        raise ValueError("cfg_scale must be at least 1")

    model_dtype = latents.dtype
    device = latents.device
    timesteps = torch.linspace(1, 0, num_steps + 1, device=device, dtype=torch.float64)
    current = latents.to(torch.float64)
    null_class = getattr(model, "num_classes", 1000)
    y_null = torch.full_like(y, null_class)

    def velocity(sample, timestep):
        use_cfg = cfg_scale > 1 and guidance_low <= timestep <= guidance_high
        model_input = torch.cat([sample, sample], dim=0) if use_cfg else sample
        labels = torch.cat([y, y_null], dim=0) if use_cfg else y
        time_input = _model_timestep(
            model_input.to(model_dtype),
            timestep,
            use_decouple_task,
            use_decouple_latent,
        )
        prediction = model(
            model_input.to(model_dtype), time_input, y=labels
        )[0].to(torch.float64)
        if use_cfg:
            conditional, unconditional = prediction.chunk(2)
            prediction = unconditional + cfg_scale * (conditional - unconditional)
        return prediction

    for index, (timestep, next_timestep) in enumerate(
        zip(timesteps[:-1], timesteps[1:])
    ):
        dt = next_timestep - timestep
        start = current
        derivative = velocity(start, timestep)
        current = start + dt * derivative
        if heun and index < num_steps - 1:
            next_derivative = velocity(current, next_timestep)
            current = start + dt * (derivative + next_derivative) / 2
    return current


@torch.no_grad()
def euler_maruyama_sampler(
    model,
    latents,
    y,
    num_steps=250,
    path_type="linear",
    condition_task=None,
    t_blend_start=0.005,
    cfg_scale=1.0,
):
    """Sample all tasks, optionally holding one 4-channel condition fixed."""
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1")
    if latents.shape[1] % 4:
        raise ValueError("Multi-task latents must contain 4 channels per task")
    num_tasks = latents.shape[1] // 4
    if condition_task is not None and not 0 <= condition_task < num_tasks:
        raise ValueError("condition_task is outside the latent task range")
    if not 0.0 <= t_blend_start <= 1.0:
        raise ValueError("t_blend_start must be between 0 and 1")
    if cfg_scale < 1.0:
        raise ValueError("cfg_scale must be at least 1")

    device = latents.device
    model_dtype = latents.dtype
    base = torch.linspace(1.0, 0.04, num_steps, device=device, dtype=torch.float64)
    base = torch.cat([base, torch.zeros(1, device=device, dtype=torch.float64)])
    timesteps = base[:, None].repeat(1, latents.shape[1])
    task_condition = None
    if condition_task is not None:
        condition_slice = slice(condition_task * 4, (condition_task + 1) * 4)
        timesteps[:, condition_slice] = t_blend_start
        task_condition = torch.full(
            (latents.shape[0],),
            condition_task + 1,
            device=device,
            dtype=torch.long,
        )

    current = latents.to(torch.float64)
    null_class = getattr(model, "num_classes", 1000)
    y_null = torch.full_like(y, null_class)
    for index, (timestep, next_timestep) in enumerate(
        zip(timesteps[:-1], timesteps[1:])
    ):
        t_cur = timestep.view(1, -1, 1, 1)
        t_next = next_timestep.view(1, -1, 1, 1)
        dt = t_next - t_cur
        time_input = t_cur.expand(current.shape[0], -1, -1, -1)
        use_cfg = cfg_scale > 1.0
        model_input = torch.cat([current, current], dim=0) if use_cfg else current
        model_time = (
            torch.cat([time_input, time_input], dim=0) if use_cfg else time_input
        )
        model_kwargs = {"y": torch.cat([y, y_null], dim=0) if use_cfg else y}
        if task_condition is not None:
            model_kwargs["lis_idx_taskcond_rand"] = (
                torch.cat([task_condition, task_condition], dim=0)
                if use_cfg
                else task_condition
            )
        prediction = model(
            model_input.to(model_dtype), model_time.to(model_dtype), **model_kwargs
        )[0].to(torch.float64)
        if use_cfg:
            conditional, unconditional = prediction.chunk(2)
            velocity = unconditional + cfg_scale * (conditional - unconditional)
        else:
            velocity = prediction
        score = get_score_from_velocity(
            velocity, current, time_input, path_type=path_type
        )
        diffusion = 2 * t_cur
        drift = velocity - 0.5 * diffusion * score
        mean = current + dt * drift
        if index < num_steps - 1:
            noise = torch.randn_like(current) * torch.sqrt(torch.abs(dt))
            current = mean + torch.sqrt(diffusion) * noise
        else:
            current = mean
    return current
