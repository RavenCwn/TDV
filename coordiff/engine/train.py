import hydra
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist
import lightning
from lightning.fabric import Fabric

import os
import wandb
import json
from tqdm import tqdm
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from coordiff.dataloader import TrajDataset, get_dataloader
from coordiff.models import *
from coordiff.utils.train_utils import setup_optimizer, setup_lr_scheduler, init_wandb
from coordiff.utils.log_utils import MetricLogger, BestAvgLoss
from diffusers.schedulers.scheduling_ddim import DDIMScheduler




@hydra.main(config_path="./config/coordiff", version_base="1.3")
def main(cfg: DictConfig):
    # Put the import here so that running on slurm does not have import error
    work_dir = HydraConfig.get().runtime.output_dir
    print(f"work_dir: {work_dir}")
    setup(cfg)
    OmegaConf.save(config=cfg, f=os.path.join(work_dir, "config.yaml"))

    train_dataset = TrajDataset(dataset_dir=cfg.train_dataset, **cfg.dataset_cfg)
    
    
    
    train_loader = get_dataloader(train_dataset,
                                    mode="train",
                                    num_workers=cfg.num_workers,
                                    batch_size=cfg.batch_size)
    
    
    val_dataset = TrajDataset(dataset_dir=cfg.val_dataset, **cfg.dataset_cfg)
    val_loader = get_dataloader(val_dataset,
                                    mode="val",
                                    num_workers=cfg.num_workers,
                                    batch_size=cfg.batch_size)



    for i, data in enumerate(train_loader):
        relevant_traj, action, task_emb, state, gripper_change = data
        print(relevant_traj.shape, action.shape, task_emb.shape, state.shape, gripper_change.shape)


        
    fabric = Fabric(accelerator="cuda", devices=list(cfg.train_gpus), precision="bf16-mixed" if cfg.mix_precision else None, strategy="deepspeed")
    fabric.launch()
    None if (cfg.dry or not fabric.is_global_zero) else init_wandb(cfg)


    # import ipdb; ipdb.set_trace()
    model_cls = eval(cfg.model_name)
    print(f"model_cls: {model_cls}")
    model = model_cls(**cfg.model_cfg)
    
    # 添加计算参数量的代码
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if fabric.is_global_zero:
        print(f"模型总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        if not cfg.dry:
            wandb.run.summary["total_parameters"] = total_params
            wandb.run.summary["trainable_parameters"] = trainable_params

    DDIM = DDIMScheduler(**cfg.ddim_cfg)

    optimizer = setup_optimizer(cfg.optimizer_cfg, model)
    scheduler = setup_lr_scheduler(optimizer, cfg.scheduler_cfg)

    
    env_num_each_rank = math.ceil(len(cfg.env_cfg.env_name) / fabric.world_size)
    env_idx_start_end = (env_num_each_rank * fabric.global_rank,  min(env_num_each_rank * (fabric.global_rank + 1), len(cfg.env_cfg.env_name)))

    fabric.barrier()
    model, optimizer = fabric.setup(model, optimizer)
    train_loader = fabric.setup_dataloaders(train_loader)

    # Pick ckpt based on  the average of the last 5 epochsenv
    metric_logger = MetricLogger(delimiter=" ")
    best_loss_logger = BestAvgLoss(window_size=5)

   
    fabric.barrier()
    for epoch in metric_logger.log_every(range(cfg.epochs), 1, ""):
        train_metrics = run_one_epoch(
            fabric,
            model,
            DDIM,
            train_loader,
            optimizer,
            cfg.clip_grad,
            mix_precision=cfg.mix_precision,
            scheduler=scheduler,
            use_vis_feat=cfg.use_vis_feat,
        )

        train_metrics["train/lr"] = optimizer.param_groups[0]["lr"]
        metric_logger.update(**train_metrics)

        if fabric.is_global_zero:
            None if cfg.dry else wandb.log(train_metrics, step=epoch)

            if epoch % cfg.val_freq == 0:
                val_metrics = evaluate(model,
                                        DDIM,
                                        val_loader,
                                        mix_precision=cfg.mix_precision,
                                        tag="val",
                                        use_vis_feat=cfg.use_vis_feat)

                # Save best checkpoint
                metric_logger.update(**val_metrics)

                val_metrics = {**val_metrics}
                loss_metric = val_metrics["val/loss"]
                is_best = best_loss_logger.update_best(loss_metric, epoch)

                if is_best:
                    model.save(f"{work_dir}/model_best.ckpt")
                    with open(f"{work_dir}/best_epoch.txt", "w") as f:
                        f.write(
                            "Best epoch: %d, Best %s: %.4f"
                            % (epoch, "loss", best_loss_logger.best_loss)
                        )
                None if cfg.dry else wandb.log(val_metrics, step=epoch)

        if epoch % cfg.save_freq == 0:
            model.save(f"{work_dir}/model_{epoch}.ckpt")

        fabric.barrier()

    if fabric.is_global_zero:
        model.save(f"{work_dir}/model_final.ckpt")
        None if cfg.dry else print(f"finished training in {wandb.run.dir}")
        None if cfg.dry else wandb.finish()


def run_one_epoch(fabric,
                  model,
                  DDIM: DDIMScheduler,
                  dataloader,
                  optimizer,
                  clip_grad=1.0,
                  use_vis_feat=False,
                  mix_precision=False,
                  scheduler=None,
                  train_timesteps = 100,
                  ):
    """
    Optimize the policy. Return a dictionary of the loss and any other metrics.
    """
    tot_loss_dict, tot_items = {}, 0

    model.train()
    i = 0
    for hist, actions_gt, text, state in tqdm(dataloader):
        if mix_precision:
            hist, actions_gt, text, state = hist.bfloat16(), actions_gt.bfloat16(), text.bfloat16(), state.bfloat16()
            

        t = torch.randint(0, train_timesteps, (hist.shape[0],), device=hist.device)

        action_noise = torch.randn_like(actions_gt)
        noicy_action = DDIM.add_noise(actions_gt, action_noise, t)

        action_noise_pred = model(noicy_action, t, hist, text=text)
        loss = F.mse_loss(action_noise_pred, action_noise, reduction="mean")

        ret_dict = {
            "loss": loss.item()
        }

        optimizer.zero_grad()
        fabric.backward(loss)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

        optimizer.step()

        for k, v in ret_dict.items():
            if k not in tot_loss_dict:
                tot_loss_dict[k] = 0
            tot_loss_dict[k] += v
        tot_items += 1

        i += 1

    out_dict = {}
    for k, v in tot_loss_dict.items():
        out_dict[f"train/{k}"] = tot_loss_dict[f"{k}"] / tot_items

    if scheduler is not None:
        scheduler.step()

    return out_dict


@torch.no_grad()
def evaluate(model, DDIM: DDIMScheduler, dataloader, mix_precision=False, tag="val", use_vis_feat=False, eval_timesteps=8):
    tot_loss_dict, tot_items = {}, 0
    model.eval()

    i = 0
    for hist, actions_gt, text, state in tqdm(dataloader):
        if mix_precision:
            hist, actions_gt, text, state = hist.bfloat16(), actions_gt.bfloat16(), text.bfloat16(), state.bfloat16()
            

        encode_cache = model.forward_enc(hist, text, state)

        # begin diffusion process
        DDIM.set_timesteps(eval_timesteps)
        DDIM.alphas_cumprod = (
            DDIM.alphas_cumprod.to(model.device)
        )

        for timestep in DDIM.timesteps:
            # predict noise given timestep
            batched_timestep = timestep.repeat(noicy_action.shape[0]).to(model.device)

            noise_pred = model.forward_dec(noicy_action, encode_cache, batched_timestep)

            action_noise_pred = noise_pred

            # take diffusion step
            noicy_action = DDIM.step(
                model_output=action_noise_pred,
                timestep=timestep,
                sample=noicy_action
            ).prev_sample

        action_loss = F.mse_loss(noicy_action, actions_gt, reduction="mean")
        loss = action_loss

        ret_dict = {
            "loss": loss.item()
        }


        i += 1

        for k, v in ret_dict.items():
            if k not in tot_loss_dict:
                tot_loss_dict[k] = 0
            tot_loss_dict[k] += v
        tot_items += 1

    out_dict = {}
    for k, v in tot_loss_dict.items():
        out_dict[f"{tag}/{k}"] = tot_loss_dict[f"{k}"] / tot_items

    return out_dict


    
def setup(cfg):
    import warnings

    warnings.simplefilter("ignore")

    lightning.seed_everything(cfg.seed)

if __name__ == "__main__":
    main()