# pytorch-dataloader-low-mem-usage

Measurements, reproduction scripts and diagnostics for the PyTorch `DataLoader`
worker memory blow-up ([pytorch/pytorch#13246](https://github.com/pytorch/pytorch/issues/13246)):
with `num_workers > 0` every worker ends up holding a private copy of the
dataset metadata, so RAM grows as `num_workers x dataset_size`.

Work in progress. See `docs/` for design notes and reviews.
