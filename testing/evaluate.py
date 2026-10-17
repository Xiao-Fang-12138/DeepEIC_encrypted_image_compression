"""Quantized DeepELIC evaluation. Run from the release root as a module."""

import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchmetrics.functional.image import structural_similarity_index_measure

from network.data import Images
from network.runtime import get_device, load_model, local_path, read_config, seed_everything


def to_uint8(x):
    # ToPILImage historically truncates after multiplying by 255; do not round.
    return (x.detach().cpu().squeeze(0).clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).numpy()


@torch.no_grad()
def evaluate(model, dataset, device, output=None, save_images=False, limit=None, padding_multiple=64):
    model.eval()  # Mandatory: enables measurement quantization.
    rows = []
    if padding_multiple < 1:
        raise ValueError("padding_multiple must be positive")
    multiple = math.lcm(padding_multiple, model.block_size)
    for index in range(len(dataset) if limit is None else min(len(dataset), limit)):
        x, name = dataset[index]
        x = x.unsqueeze(0).to(device)
        h, w = x.shape[-2:]
        if min(h, w) < 11:
            raise ValueError(f"{name}: evaluation images must be at least 11 pixels in each dimension for SSIM")
        ph, pw = (-h) % multiple, (-w) % multiple
        top, left = ph // 2, pw // 2
        padding = (left, pw - left, top, ph - top)
        mode = "reflect" if max(top, ph - top) < h and max(left, pw - left) < w else "replicate"
        padded = F.pad(x, padding, mode=mode)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = model(padded)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        assert result["quantized"] and result["symbols"].dtype == torch.uint8
        reconstruction = result["x_hat"][:, :, top:top + h, left:left + w]
        # Historical SSIM: float RGB before clamping. PSNR: clamped uint8 RGB.
        ssim = structural_similarity_index_measure(reconstruction, x, data_range=1.0).item()
        target, restored = to_uint8(x), to_uint8(reconstruction)
        mse = float(np.mean((target.astype(np.float64) - restored.astype(np.float64))**2))
        psnr = 10 * math.log10(255**2 / mse) if mse else float("inf")
        float_mse = F.mse_loss(reconstruction, x).item()
        row = dict(image=name, psnr=psnr, ssim=ssim, mse=float_mse, seconds=elapsed)
        rows.append(row)
        if save_images:
            destination = output / "images" / (name + ".png")
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(restored).save(destination)
        print(f"[{index + 1}/{len(dataset)}] {name}: PSNR {psnr:.4f}, SSIM {ssim:.6f}", flush=True)
    summary = {key: sum(row[key] for row in rows) / len(rows) for key in ("psnr", "ssim", "mse", "seconds")}
    return dict(count=len(rows), average=summary, images=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/x2.json")
    parser.add_argument("--checkpoint", help="Override the configuration checkpoint (relative path)")
    parser.add_argument("--data", default="data/test/DIV2K")
    parser.add_argument("--output", help="Relative output directory; defaults to results/<task>")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--crop", type=int, help="Center crop size; 0 evaluates the complete image")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--save-images", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.crop is not None and args.crop < 0:
        parser.error("--crop must be nonnegative")
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    config = read_config(args.config)
    checkpoint = args.checkpoint or config["checkpoint"]
    device = get_device(args.device)
    model = load_model(config, checkpoint, device)
    crop = config["evaluation"]["crop"] if args.crop is None else args.crop
    dataset = Images(args.data, crop=crop)
    output = local_path(args.output or f"results/{config['task']}")
    output.mkdir(parents=True, exist_ok=True)
    report = evaluate(model, dataset, device, output, args.save_images, args.limit,
                      padding_multiple=config["evaluation"]["padding_multiple"])
    report.update(config=config["model"], checkpoint=checkpoint, dataset=args.data, crop=crop, seed=args.seed,
                  padding_multiple=config["evaluation"]["padding_multiple"],
                  quantization="per-image 8-bit min/max", torch_version=str(torch.__version__), device=str(device))
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["average"], indent=2))


if __name__ == "__main__":
    main()
