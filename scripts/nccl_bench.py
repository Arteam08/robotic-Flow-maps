"""Time a SiT-XL/2-sized fp32 all-reduce (675M params = 2.7 GB) under torchrun.
Usage: python -m torch.distributed.run --standalone --nproc_per_node=N scripts/nccl_bench.py [numel]
"""
import os, sys, time, torch, torch.distributed as dist
numel = int(float(sys.argv[1])) if len(sys.argv) > 1 else 675_000_000
dist.init_process_group("nccl")
r = dist.get_rank(); dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", r))); torch.cuda.set_device(dev)
x = torch.ones(numel, device=dev)
for _ in range(2): dist.all_reduce(x)
torch.cuda.synchronize(); dist.barrier()
t = time.time(); n = 5
for _ in range(n): dist.all_reduce(x)
torch.cuda.synchronize(); dt = (time.time() - t) / n
xb = x.to(torch.bfloat16)
for _ in range(2): dist.all_reduce(xb)
torch.cuda.synchronize(); t = time.time()
for _ in range(n): dist.all_reduce(xb)
torch.cuda.synchronize(); dtb = (time.time() - t) / n
if r == 0:
    gb = numel * 4 / 1e9
    print(f"world={dist.get_world_size()} P2P_DISABLE={os.environ.get('NCCL_P2P_DISABLE','unset')} "
          f"fp32 {gb:.2f} GB all-reduce: {dt:.3f} s ({2*(dist.get_world_size()-1)/dist.get_world_size()*gb/dt:.1f} GB/s busbw) | "
          f"bf16: {dtb:.3f} s", flush=True)
dist.destroy_process_group()
