# PyTorch Distributed Training Command Format

## Basic Structure
```bash
python -m torch.distributed.launch \
    --nproc_per_node=<num_gpus> \
    --master_addr=<master_address> \
    --master_port=<port_number> \
    <training_script.py> \
    --config=<config_file_path> \
    --port <port_number> \
    --resume-from <checkpoint_path>
```

## Parameter Descriptions

### Distributed Launch Parameters
- `--nproc_per_node`: Number of processes per node (typically equals number of GPUs)
- `--master_addr`: IP address of the master node (use `localhost` for single machine)
- `--master_port`: Port for master process communication

### Training Script Parameters
- `<training_script.py>`: Your main training script
- `--config`: Path to configuration file (YAML/JSON)
- `--port`: Port for the training process
- `--resume-from`: Path to checkpoint file for resuming training

## Example Usage
```bash
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_addr=localhost \
    --master_port=12345 \
    semivl.py \
    --config=configs/experiment/config.yaml \
    --port 12345 \
    --resume-from exp/checkpoints/latest.pth
```

## Common Variations
- Multi-GPU: Set `--nproc_per_node=4` for 4 GPUs
- Multi-node: Use actual IP addresses for `--master_addr`
- Fresh training: Remove `--resume-from` parameter

## TensorBoard Visualization
```bash
# Start TensorBoard to view training logs
tensorboard --logdir=<log_directory>

# Example
tensorboard --logdir=exp/exp/mmseg.vlm-dlv3p-bn12-sk4-ftap-mcvitb_brickfield_bs1_250607-1327_61988
```
Then open `http://localhost:6006` in your browser to view training metrics, loss curves, and other logged data.