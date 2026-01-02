import itertools as it
import threading, time
import torch
from typing import Optional
from torch.utils.data import Sampler, IterableDataset, DataLoader
from transformers.trainer_utils import has_length
from .logging_trainer import TrainerWithModelMetrics
from datasets import IterableDataset as HFIterableDataset
import logging

logger = logging.getLogger(__name__)


class OffsetRandomSampler(Sampler):
    def __init__(self, data_source, offset: int, seed: Optional[int] = None):
        self.data_source = data_source
        self.offset = max(0, int(offset))
        self.seed = seed
        self._consumed = False

    def __iter__(self):
        n = len(self.data_source)
        if self.seed is None:
            perm = torch.randperm(n).tolist()
        else:
            g = torch.Generator()
            g.manual_seed(int(self.seed))
            perm = torch.randperm(n, generator=g).tolist()
        off = self.offset if not self._consumed else 0
        self._consumed = True
        for i in perm[off:]:
            yield i

    def __len__(self):
        return len(self.data_source)


class _SkipIterableOnce(IterableDataset):
    def __init__(self, base: IterableDataset, n_to_skip: int,
                 log_every: int = 0):
        self.base = base
        self.n_to_skip = max(0, int(n_to_skip))
        self.log_every = int(log_every)
        self._done = False
        self._lock = threading.Lock()
        self._worker_done = {}
        self._is_hf_iter = (
            HFIterableDataset is not None and isinstance(
                base, HFIterableDataset
            )
        )
        flag = bool(self._is_hf_iter)
        logger.info(f"data is HuggingFace IterableDataset: {flag}")

    def __iter__(self):
        itb = iter(self.base)
        info = torch.utils.data.get_worker_info()
        if info is None:
            wid = -1
            local_skip = self.n_to_skip
        else:
            wid = info.id
            per = self.n_to_skip // info.num_workers
            rem = self.n_to_skip % info.num_workers
            local_skip = per + (1 if wid < rem else 0)

        with self._lock:
            if self._done and self._worker_done.get(wid, False):
                local_skip = 0
            else:
                self._worker_done[wid] = True

        if local_skip > 0 and self._is_hf_iter:
            logger.info(
                    f"[skip] worker {wid}: skipping {local_skip} local "
                    "samples..."
                )
            itb = iter(self.base.skip(local_skip))
            local_skip = 0
        elif local_skip > 0:
            if self.log_every:
                logger.info(
                    f"[skip] worker {wid}: skipping {local_skip} samples..."
                )
            skipped = 0
            t0 = time.time()
            chunk = self.log_every if self.log_every else local_skip
            while skipped < local_skip:
                step = min(chunk, local_skip - skipped)
                for _ in it.islice(itb, step):
                    pass
                skipped += step
                if self.log_every:
                    dt = time.time() - t0
                    frac = skipped / max(local_skip, 1)
                    logger.info(
                        f"[skip] worker {wid}: {skipped}/{local_skip} "
                        f"({100.0 * frac:.1f}%) in {dt:.1f}s"
                    )

        with self._lock:
            self._done = True

        for x in itb:
            yield x


class TrainerDataSkipping(TrainerWithModelMetrics):
    def __init__(self, *args, skip_train_batches: Optional[int] = 0,
                 skip_log_every: int = 100000, **kwargs):
        super().__init__(*args, **kwargs)
        if skip_train_batches is None:
            skip_train_batches = 0

        bsz = getattr(self.args, "per_device_train_batch_size", 1)
        ga = max(1, getattr(self.args, "gradient_accumulation_steps", 1))
        self._skip_local_samples = int(skip_train_batches) * int(bsz) * int(ga)
        self._skip_log_every = int(skip_log_every)

        if getattr(self.args, "local_rank", -1) in (-1, 0):
            msg = (
                f"[skip] plan: {int(skip_train_batches)} global steps -> "
                f"{int(self._skip_local_samples)} local samples "
                f"(bsz={int(bsz)}, ga={int(ga)})"
            )
            logger.info(msg)

    def _get_train_sampler_with_skipping(self, train_dataset=None):
        ds = train_dataset or self.train_dataset
        if ds is None or not has_length(ds):
            return None
        if self.args.group_by_length:
            raise ValueError(
                f"set group_by_length=False with OffsetRandomSampler"
            )
        if self._skip_local_samples > 0:
            seed = getattr(self.args, "data_seed", None)
            if seed is None:
                seed = getattr(self.args, "seed", None)
            return OffsetRandomSampler(
                ds, offset=self._skip_local_samples, seed=seed
            )
        seed = getattr(self.args, "data_seed", None)
        if seed is None:
            seed = getattr(self.args, "seed", None)
        if seed is not None:
            g = torch.Generator()
            g.manual_seed(int(seed))
            return torch.utils.data.RandomSampler(ds, generator=g)
        return torch.utils.data.RandomSampler(ds)

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError(f"Trainer: training requires a train_dataset.")

        ds = self.train_dataset
        if isinstance(ds, IterableDataset):
            if self._skip_local_samples > 0:
                ds = _SkipIterableOnce(
                    ds, self._skip_local_samples, self._skip_log_every
                )
            return self._get_dataloader(
                dataset=ds,
                description=f"Training",
                batch_size=self._train_batch_size,
                sampler_fn=self._get_train_sampler,
                is_training=True,
            )

        return self._get_dataloader(
            dataset=ds,
            description=f"Training",
            batch_size=self._train_batch_size,
            sampler_fn=self._get_train_sampler_with_skipping,
            is_training=True,
        )