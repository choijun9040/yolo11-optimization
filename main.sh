GPUS=$1
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=$GPUS main.py ${@:2}