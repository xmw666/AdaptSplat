

import importlib
import os
import warnings
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler

from setup import init_config, init_distributed
from metric_utils import export_results, summarize_evaluation


class NoPaddingDistributedSampler(Sampler):
    """分布式采样器，不填充样本，保证每个样本只被推理一次。"""

    def __init__(self, dataset, num_replicas, rank):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.total_size = len(dataset)
        self.num_samples = (self.total_size + self.num_replicas - 1 - self.rank) // self.num_replicas

    def __iter__(self):
        # 按间隔分配：rank 0 取 0, 8, 16...; rank 1 取 1, 9, 17... 等
        indices = list(range(self.rank, self.total_size, self.num_replicas))
        return iter(indices)

    def __len__(self):
        return self.num_samples


def main():
    # ------------------------------------------------------------------ #
    # 1. 初始化分布式环境（每个 rank 对应一张 GPU）
    # ------------------------------------------------------------------ #
    dist_info = init_distributed()
    rank       = dist_info.global_rank
    world_size = dist_info.world_size
    device     = dist_info.device
    is_main    = dist_info.is_main_process

    # ------------------------------------------------------------------ #
    # 2. 加载配置（所有 rank 读同一份 yaml，CLI override 同样生效）
    # ------------------------------------------------------------------ #
    config = init_config()

    os.environ["OMP_NUM_THREADS"] = str(config.inference.get("num_threads", 1))

    torch.backends.cuda.matmul.allow_tf32 = config.inference.use_tf32
    torch.backends.cudnn.allow_tf32       = config.inference.use_tf32

    amp_dtype_mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
        "tf32": torch.float32,
    }

    # ------------------------------------------------------------------ #
    # 3. 构建数据集 + NoPaddingDistributedSampler（按 rank 间隔分配，不重复不遗漏）
    # ------------------------------------------------------------------ #
    dataset_name = config.inference.get("dataset_name", "dataset.Dataset")
    module_name, class_name = dataset_name.rsplit(".", 1)
    DatasetClass = importlib.import_module(module_name).__dict__[class_name]
    dataset = DatasetClass(config)

    sampler = NoPaddingDistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.inference.batch_size_per_gpu,
        sampler=sampler,
        num_workers=config.inference.num_workers,
        prefetch_factor=config.inference.prefetch_factor,
        persistent_workers=True,
        pin_memory=False,
    )

    # ------------------------------------------------------------------ #
    # 4. 加载模型（每个 rank 在自己的 GPU 上独立加载一份权重）
    # ------------------------------------------------------------------ #
    module_name, class_name = config.model.class_name.rsplit(".", 1)
    MVP = importlib.import_module(module_name).__dict__[class_name]
    model = MVP(config).to(device)
    model.load_ckpt(config.inference.ckpt_path)
    model.eval()

    if is_main:
        print(f"[DDP Inference] world_size={world_size}")
        print(f"[DDP Inference] dataset size={len(dataset)}, "
              f"per-rank≈{len(sampler)} scenes")
        print(f"[DDP Inference] Saving results to: {config.inference.out_dir}")

    warnings.filterwarnings("ignore", category=FutureWarning)

    # ------------------------------------------------------------------ #
    # 5. 推理循环
    #    uid = rank + local_cnt * world_size，真实数据集索引
    # ------------------------------------------------------------------ #
    out_dir  = config.inference.out_dir
    num_input_frames = config.data.num_input_frames

    local_cnt = 0
    with torch.no_grad(), torch.autocast(
        enabled=config.inference.use_amp,
        device_type="cuda",
        dtype=amp_dtype_mapping[config.inference.amp_dtype],
    ):
        for batch in dataloader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            input_data_dict = {
                k: v[:, :num_input_frames] if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            target_data_dict = {
                k: v[:, num_input_frames:] if isinstance(v, torch.Tensor) else None
                for k, v in batch.items()
            }

            result = model(input_data_dict, target_data_dict)

            # uid 为真实数据集索引（间隔分配：rank + local_cnt * world_size）
            uid = rank + local_cnt * world_size
            export_results(
                result,
                out_dir,
                compute_metrics=config.inference.get("compute_metrics", False),
                uid=uid,
            )

            if is_main:
                print(f"[rank {rank}] processed scene {local_cnt + 1}/{len(sampler)}")
            local_cnt += 1

        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # 6. 等待所有 rank 完成后，rank 0 汇总指标
    # ------------------------------------------------------------------ #
    dist.barrier()

    if is_main and config.inference.get("compute_metrics", False):
        summarize_evaluation(out_dir)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
