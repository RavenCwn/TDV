import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import swanlab
from tqdm import tqdm
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from coordiff.dataloader import TrajDataset, get_dataloader
from coordiff.models import *
from coordiff.utils.train_utils import setup_optimizer, setup_lr_scheduler, init_swanlab
from coordiff.utils.log_utils import MetricLogger, BestAvgLoss
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

@hydra.main(config_path="../config", version_base="1.3")
def main(cfg: DictConfig):
    # Put the import here so that running on slurm does not have import error
    # work_dir = HydraConfig.get().runtime.output_dir
    work_dir = cfg.work_dir
    print(f"work_dir: {work_dir}")
    setup(cfg)
    OmegaConf.save(config=cfg, f=os.path.join(work_dir, "config.yaml"))

    # 设置设备
    device = torch.device(f"cuda:{cfg.train_gpus[0]}" if torch.cuda.is_available() else "cpu")
    
    train_dataset = TrajDataset(
        dataset_dir=cfg.train_dataset,
        mode='train',
        task_list=cfg.task_list,
        **cfg.dataset_cfg
    )
    val_dataset = TrajDataset(
        dataset_dir=cfg.val_dataset,
        mode='train',
        task_list=cfg.task_list,
        **cfg.dataset_cfg)
    
    train_loader = get_dataloader(train_dataset,
                                mode="train",
                                num_workers=cfg.num_workers,
                                batch_size=cfg.batch_size)
    
    val_loader = get_dataloader(val_dataset,
                                mode="train",
                                num_workers=cfg.num_workers,
                                batch_size=cfg.batch_size)
        

    # for i, data in enumerate(train_loader):
    #     print(i)
    #     hist, action, task_emb, state, gripper_change = data
    #     print(hist.shape, action.shape, task_emb.shape, state.shape, gripper_change.shape)

    if not cfg.dry:
        init_swanlab(cfg)

    model_cls = eval(cfg.arm_model_name)
    arm_model = model_cls(**cfg.arm_model_cfg, device=device).to(device)
    
    print(f"model name: {cfg.arm_model_name}")
    total_params = sum(p.numel() for p in arm_model.parameters())
    trainable_params = sum(p.numel() for p in arm_model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    if not cfg.dry:
        # swanlab.summary["arm_total_parameters"] = total_params
        # swanlab.summary["arm_trainable_parameters"] = trainable_params
        swanlab.log({"arm_total_parameters": total_params})
        swanlab.log({"arm_trainable_parameters": trainable_params})


    model_cls = eval(cfg.gripper_model_name)
    gripper_model = model_cls(**cfg.gripper_model_cfg, device=device).to(device)
    print(f"model name: {cfg.gripper_model_name}")
    total_params = sum(p.numel() for p in gripper_model.parameters())
    trainable_params = sum(p.numel() for p in gripper_model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    if not cfg.dry:
        # swanlab.summary["gripper_total_parameters"] = total_params
        # swanlab.summary["gripper_trainable_parameters"] = trainable_params
        swanlab.log({"gripper_total_parameters": total_params})
        swanlab.log({"gripper_trainable_parameters": trainable_params})

    if hasattr(cfg, "resume_path"):
        arm_model.load_state_dict(torch.load(cfg.resume_path + '/arm_model_last.ckpt'))
        gripper_model.load_state_dict(torch.load(cfg.resume_path + '/gripper_model_last.ckpt'))
        print("Successfully resume from: ", cfg.resume_path)

    DDIM = DDIMScheduler(**cfg.ddim_cfg)
    arm_optimizer = setup_optimizer(cfg.optimizer_cfg, arm_model)
    arm_scheduler = setup_lr_scheduler(arm_optimizer, cfg.scheduler_cfg)

    gripper_optimizer = setup_optimizer(cfg.optimizer_cfg, gripper_model)
    gripper_scheduler = setup_lr_scheduler(gripper_optimizer, cfg.scheduler_cfg)


    # 训练循环
    metric_logger = MetricLogger(delimiter=" ")
    best_loss_logger = BestAvgLoss(window_size=5)
    
    for epoch in metric_logger.log_every(range(cfg.epochs), 1, ""):
        train_metrics = run_one_epoch(
            model=(arm_model, gripper_model),
            DDIM=DDIM,
            dataloader=train_loader,
            optimizer=(arm_optimizer, gripper_optimizer),
            device=device,
            clip_grad=cfg.clip_grad,
            mix_precision=cfg.mix_precision,
            scheduler=(arm_scheduler, gripper_scheduler),
        )

        train_metrics["train/lr"] = arm_optimizer.param_groups[0]["lr"]
        metric_logger.update(**train_metrics)

        if not cfg.dry:
            swanlab.log(train_metrics, step=epoch)

        if epoch % cfg.val_freq == 0:
            val_metrics = evaluate(
                model=(arm_model, gripper_model),
                DDIM=DDIM,
                dataloader=val_loader,
                device=device,
                mix_precision=cfg.mix_precision,
                tag="val",
                eval_timesteps=cfg.eval_timesteps
            )

            metric_logger.update(**val_metrics)
            loss_metric = val_metrics["val/loss"]
            is_best = best_loss_logger.update_best(loss_metric, epoch)

            if is_best:
                torch.save(arm_model.state_dict(), f"{work_dir}/arm_model_best.ckpt")
                torch.save(gripper_model.state_dict(), f"{work_dir}/gripper_model_best.ckpt")
                with open(f"{work_dir}/best_epoch.txt", "w") as f:
                    f.write(
                        "Best epoch: %d, Best %s: %.4f"
                        % (epoch, "loss", best_loss_logger.best_loss)
                    )
            if not cfg.dry:
                swanlab.log(val_metrics, step=epoch)

        if epoch % cfg.save_freq == 0:
            torch.save(arm_model.state_dict(), f"{work_dir}/arm_model_{epoch}.ckpt")
            torch.save(gripper_model.state_dict(), f"{work_dir}/gripper_model_{epoch}.ckpt")
            torch.save(arm_model.state_dict(), f"{work_dir}/arm_model_last.ckpt")
            torch.save(gripper_model.state_dict(), f"{work_dir}/gripper_model_last.ckpt")

    torch.save(arm_model.state_dict(), f"{work_dir}/arm_model_final.ckpt")
    torch.save(arm_model.state_dict(), f"{work_dir}/gripper_model_final.ckpt")



def run_one_epoch(model,
                  DDIM: DDIMScheduler,
                  dataloader,
                  optimizer,
                  device,
                  clip_grad=1.0,
                  mix_precision=False,
                  scheduler=None,
                  train_timesteps = 100,
                  ):
    tot_loss_dict, tot_items = {}, 0
    arm_model, gripper_model = model
    arm_optimizer, gripper_optimizer = optimizer
    arm_scheduler, gripper_scheduler = scheduler
    arm_model.train()
    gripper_model.train()
    
    weights = torch.tensor([0.2, 0.8]).to(device)  # 示例权重

    for hist, actions_gt, task_emb, state, gripper_change in tqdm(dataloader):
        hist = hist.to(device)
        actions_gt = actions_gt.to(device)
        task_emb = task_emb.to(device)
        state = state.to(device)
        gripper_change = gripper_change.to(device)
        
        if mix_precision:
            hist, action, task_emb, state, gripper_change = hist.bfloat(), action.bfloat(), task_emb.bfloat(), state.bfloat(), gripper_change.bfloat()

        t = torch.randint(0, train_timesteps, (hist.shape[0],), device=device)
        action_noise = torch.randn_like(actions_gt)
        noicy_action = DDIM.add_noise(actions_gt, action_noise, t)

        action_noise_pred = arm_model(noicy_action, t, hist, text=task_emb, state=state)
        arm_loss = F.mse_loss(action_noise_pred, action_noise, reduction="mean")

        arm_optimizer.zero_grad()
        arm_loss.backward()
        torch.nn.utils.clip_grad_norm_(arm_model.parameters(), max_norm=clip_grad)
        arm_optimizer.step()


        state_change = gripper_model(hist, task_emb, state)
        gripper_loss = F.cross_entropy(state_change, gripper_change, weight=weights, reduction="mean")
        gripper_optimizer.zero_grad()
        gripper_loss.backward()
        torch.nn.utils.clip_grad_norm_(gripper_model.parameters(), max_norm=clip_grad)
        gripper_optimizer.step()


        ret_dict = {
            "loss": arm_loss.item()+gripper_loss.item(),
            "arm_loss": arm_loss.item(),
            "gripper_loss": gripper_loss.item()
        }
        
        for k, v in ret_dict.items():
            if k not in tot_loss_dict:
                tot_loss_dict[k] = 0
            tot_loss_dict[k] += v
        tot_items += 1

    out_dict = {}
    for k, v in tot_loss_dict.items():
        out_dict[f"train/{k}"] = tot_loss_dict[f"{k}"] / tot_items

    if arm_scheduler is not None:
        arm_scheduler.step()

    return out_dict


@torch.no_grad()
def evaluate(model, DDIM: DDIMScheduler, dataloader, device, mix_precision=False, tag="val", eval_timesteps=8):
    tot_loss_dict, tot_items = {}, 0
    arm_model, gripper_model = model
    arm_model.eval()
    gripper_model.eval()

    weights = torch.tensor([0.2, 0.8]).to(device)  # 示例权重
    i = 0
    for hist, actions_gt, task_emb, state, gripper_change in tqdm(dataloader):
        hist = hist.to(device)
        actions_gt = actions_gt.to(device)
        task_emb = task_emb.to(device)
        state = state.to(device)
        gripper_change = gripper_change.to(device)
        
        if mix_precision:
            hist, action, task_emb, state, gripper_change = hist.bfloat(), action.bfloat(), task_emb.bfloat(), state.bfloat(), gripper_change.bfloat()

        encode_cache = arm_model.forward_enc(hist, task_emb, state)

        # begin diffusion process
        DDIM.set_timesteps(eval_timesteps)
        DDIM.alphas_cumprod = (
            DDIM.alphas_cumprod.to(device)
        )

        noicy_action = torch.randn_like(actions_gt)

        for timestep in DDIM.timesteps:
            # predict noise given timestep
            batched_timestep = timestep.repeat(noicy_action.shape[0]).to(device)

            noise_pred = arm_model.forward_dec(noicy_action, encode_cache, batched_timestep)

            action_noise_pred = noise_pred

            # take diffusion step
            noicy_action = DDIM.step(
                model_output=action_noise_pred,
                timestep=timestep,
                sample=noicy_action
            ).prev_sample

        arm_loss = F.mse_loss(noicy_action, actions_gt, reduction="mean")
        
        state_change = gripper_model(hist, task_emb, state)
        gripper_loss = F.cross_entropy(state_change, gripper_change, reduction="mean", weight=weights)
        state_change = state_change.argmax(dim=-1)
        acc = (state_change == gripper_change).float().mean()


        ret_dict = {
            "loss": arm_loss.item()+gripper_loss.item(),
            "arm_loss": arm_loss.item(),
            "gripper_loss": gripper_loss.item(),
            "gripper_acc": acc.item()
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


if __name__ == "__main__":
    main()