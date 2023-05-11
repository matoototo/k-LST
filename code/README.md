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
# TensorBoard
```bash
tensorboard --logdir results/runs
```