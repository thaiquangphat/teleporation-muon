# teleport

## Training logs

`src/train.py` writes a uniquely named `training_log_<run-id>.json` into the
same run directory as `last.pt` and `best.pt`. The log is updated after every
completed epoch and records the run configuration, device and dataset details,
training and validation loss, validation accuracy, learning rates,
per-epoch timings, total training time, and the best validation result. If
training raises an error, the log is marked as failed and includes the error
details and metrics from the completed epochs.