# Setup
```bash
# Create virtual environment
python3 -m venv .env
# Activate virtual environment
source .env/bin/activate
# Install dependencies
pip install -r requirements.txt
# Deactivate virtual environment when finished
deactivate
```
# Usage
Start TensorBoard
```bash
tensorboard --logdir results/runs
```
Run training with config file
```bash
python3 ./train_utils.py --config configs/example.yaml
```

### Options

| Option                                            | Description                                                                                                                              |
|---------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| --config <config_file>                            | Path to config file. Required.                                                                                                           |
| --model_path <model_path>                         | Path to local checkpoint directory to initialize the model from.                                                                         |
| --resume_from_checkpoint <resume_from_checkpoint> | When set to True, trainer resumes from latest checkpoint. When set to a path to a checkpoint, trainer resumes from the given checkpoint. |
