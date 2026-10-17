"""Train DeepELIC without quantization; validate with 8-bit quantization."""

import argparse
import json
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from network.data import Images
from network.runtime import get_device, load_model, local_path, read_config, seed_everything
from testing.evaluate import evaluate


def save_checkpoint(path, model, optimizer, scheduler, step, best_loss):
    state = dict(format_version=1, model_config=model.config, state_dict=model.state_dict(),
                 optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), step=step,
                 best_loss=best_loss, torch_rng=torch.get_rng_state(), python_rng=random.getstate())
    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/x2.json")
    parser.add_argument("--data", default="data/train/DIV2K")
    parser.add_argument("--val-data", default="data/test/DIV2K")
    parser.add_argument("--output", help="Relative output directory; defaults to training/runs/<task>")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    for name in ("max-steps", "batch-size", "patch-size", "val-every", "save-every"):
        parser.add_argument("--" + name, type=int)
    parser.add_argument("--learning-rate", type=float)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--init", help="Initialize weights only; use this for released checkpoints")
    initialization.add_argument("--resume", help="Resume a checkpoint produced by this training script")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    config = read_config(args.config)
    settings = dict(config["training"])
    for key in settings:
        if getattr(args, key, None) is not None:
            settings[key] = getattr(args, key)
    if min(settings[k] for k in ("max_steps", "batch_size", "patch_size", "save_every")) < 1 or settings["val_every"] < 0:
        parser.error("Training sizes and save frequency must be positive; val-every may be zero")
    if settings["patch_size"] % config["model"]["block_size"]:
        parser.error("patch-size must be divisible by block_size")
    device = get_device(args.device)
    model = load_model(config, args.init or args.resume, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"], betas=(0.9, 0.999), weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=settings["lr_step"], gamma=settings["lr_gamma"])
    step, best_loss = 0, float("inf")
    if args.resume:
        state = torch.load(local_path(args.resume), map_location="cpu", weights_only=True)
        if "optimizer" not in state or "step" not in state:
            parser.error("Released weights have no optimizer; use --init instead of --resume")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step, best_loss = state["step"], state["best_loss"]
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        if device.type == "cuda" and "cuda_rng" in state:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    dataset = Images(args.data, crop=settings["patch_size"], training=True)
    loader = DataLoader(dataset, batch_size=settings["batch_size"], shuffle=True, num_workers=args.workers,
                        pin_memory=device.type == "cuda", drop_last=False)
    validation = Images(args.val_data, crop=config["evaluation"]["crop"]) if settings["val_every"] else None
    output = local_path(args.output or f"training/runs/{config['task']}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(dict(model=config["model"], training=settings, arguments=vars(args)), indent=2) + "\n")
    with (output / "train.jsonl").open("a") as log:
        while step < settings["max_steps"]:
            for x, _ in loader:
                model.train()  # Always bypass measurement quantization during optimization.
                x = x.to(device)
                optimizer.zero_grad(set_to_none=True)
                result = model(x)
                assert not result["quantized"]
                loss = 255**2 * F.mse_loss(result["x_hat"], x)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                loss.backward()
                optimizer.step()
                scheduler.step()
                step += 1
                row = dict(step=step, loss=loss.item(), learning_rate=optimizer.param_groups[0]["lr"])
                if validation is not None and step % settings["val_every"] == 0:
                    report = evaluate(model, validation, device, padding_multiple=config["evaluation"]["padding_multiple"])
                    row["validation"] = report["average"]
                    validation_loss = 255**2 * report["average"]["mse"]
                    if validation_loss < best_loss:
                        best_loss = validation_loss
                        save_checkpoint(output / "best.pth", model, optimizer, scheduler, step, best_loss)
                if step % settings["save_every"] == 0 or step == settings["max_steps"]:
                    save_checkpoint(output / "latest.pth", model, optimizer, scheduler, step, best_loss)
                log.write(json.dumps(row) + "\n")
                log.flush()
                if step == 1 or step % 10 == 0 or step == settings["max_steps"]:
                    print(json.dumps(row), flush=True)
                if step >= settings["max_steps"]:
                    break


if __name__ == "__main__":
    main()
