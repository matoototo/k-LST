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
python3 ./train_utils.py --config example.yaml
```