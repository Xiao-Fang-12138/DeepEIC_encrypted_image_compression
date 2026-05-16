# DeepELIC: Official PyTorch Implementation

This repository contains the official PyTorch implementation for the paper **DeepELIC**.

> **📢 Notice:** The pre-trained checkpoints and models will be made publicly available immediately upon the acceptance of the paper.

## 🛠️ Dependencies

Ensure your environment meets the following requirements:

- Python >= 3.8
- PyTorch
- compressai
- torchvision
- torchmetrics
- thop
- tqdm

## 🚀 Getting Started

### 1. Training

To train the DeepELIC model from scratch, use the `train.py` script. You need to specify the paths for the training and testing datasets, as well as an experiment name.

```bash
python train.py -d /path/to/train_dataset -d_test /path/to/test_dataset -exp my_experiment --cuda
```

**Key Arguments:**

- `--batch-size`: Batch size for training (Default: `16`).
- `-e`, `--epochs`: Number of training epochs (Default: `100000`).
- `-lr`, `--learning-rate`: Learning rate (Default: `1e-4`).
- `--checkpoint`: Path to resume training from a specific checkpoint.

**Outputs:**

- **Logs**: Saved in `experiments/<experiment_name>/` and TensorBoard logs in `./tb_logger/<experiment_name>`.
- **Checkpoints**: Saved in `experiments/<experiment_name>/checkpoints/`. The best model will be saved as `net_checkpoint_best_loss.pth.tar`.

### 2. Testing & Evaluation

To evaluate the model, use the `test.py` script. **Please note that the official pre-trained checkpoints will be provided upon paper acceptance.** Once available, you can run the evaluation using the `--test` flag and providing the checkpoint path.

```bash
python test.py -d /path/to/dummy_train -d_test /path/to/test_dataset -exp test_experiment --checkpoint /path/to/net_checkpoint_best_loss.pth.tar --cuda --test
```

*(Note: The* *`-d`* *argument is required for argument parsing, though it is not actively used during the testing phase.)*

**Outputs:**

- **Metrics**: Evaluation results (PSNR, SSIM, Bpp loss, etc.) will be logged in `experiments/<experiment_name>/`.
- **Visualizations**: Reconstructed images and intermediate encrypted results are saved in `experiments/<experiment_name>/images/`:
  - `ori/`: Original high-resolution images.
  - `rec/`: Reconstructed images.
  - `final_enc/`: Final encrypted images.

