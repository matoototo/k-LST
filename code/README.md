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
Additionally, to enable the profiler you can pass the --profile flag:
```bash
python3 ./train_utils.py --config configs/example.yaml --profile
```

The profiling stack trace will be under root/trace.json. You should also limit the dataset size in the config when profiling to save time. A couple batches is more than enough.