# Business entity resolution

Implementation status: code and synthetic checks only. No model has been trained.

## Ubuntu: one-command full training

Copy this project and the supplied `student_resource` folder onto the server, then:

```bash
bash run_training.sh --data /path/to/student_resource
```

For the NVIDIA A30 server, explicitly require CUDA:

```bash
bash run_training.sh --data /path/to/student_resource --device cuda
```

The default `auto` mode selects CUDA when a usable NVIDIA GPU is detected; `--device
cuda` makes missing GPU support an error. CUDA runs use `runs/full-training-cuda/`
and a separate Python environment. The GPU accelerates tree training; retrieval,
feature generation, the Python F0.5 metric, and exported-model prediction use the CPU.

The launcher installs software dependencies, creates an isolated Python 3.12
environment, checks the code, indexes the complete training dataset, prepares
features, trains and compares LightGBM models, selects the F0.5 threshold, evaluates
the untouched audit split, and saves the final model. It prints progress in the
terminal and ends with **TRAINING COMPLETE** and **FULL PIPELINE COMPLETED SUCCESSFULLY**
only after the corresponding stages succeed.

If `student_resource/` is beside the script, simply run `bash run_training.sh`.
See [UBUNTU_TRAINING.md](UBUNTU_TRAINING.md) for server requirements, output paths,
restart behavior, and optional test prediction.

- `code/business_entity_resolution/`: runnable pipeline, configuration, and tests.
- `analysis/scripts/`: exploratory scripts from the initial dataset examination.
- `analysis/reports/`: original measured dataset profiles.
- `output/`: reserved for final matching and candidate TSVs; currently empty.
- `run_training.sh`: Ubuntu dependency setup and full-pipeline launcher.
- `runs/full-training/`: default server run directory (created only when launched).
- `Documentation_template.md`: methodology and explicitly unmeasured results.

The local dataset remains in the original student_resource folder. Do not commit
the large TSVs to Git. See the pipeline README for stage commands and limitations.
