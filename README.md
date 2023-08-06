# Lab Course 23-01 Reducing Training Resource Cost by Selective Parameter Updating

**Authors:** Baykam Say, Barış Coşlu, and Mato Gudelj \
**Supervisor:** M.Sc. Jeremias Bohn

This work introduces two novel Parameter-Efficient Fine Tuning (PEFT) methods, Singular Value Adaptation (SiVA) and k-Ladder Side-Tuning (k-LST), for fine-tuning of large-scale pre-trained transformer models. SiVA improves upon LoRA by leveraging singular value decomposition for initialization, avoiding the problematic zero-initialization employed by LoRA. This leads to faster convergence while maintaining full fine-tuning performance. k-LST improves upon Ladder Side Tuning, closing the performance gap of memory efficient PEFT methods. It extracts backbone features with a sliding window, which are then queried by the side network with cross-attention, while retaining a low memory footprint. We additionally utilized prompts during fine-tuning. We observed a significant improvement with no computational drawbacks, significantly improving performance over the baseline.

## Repository Overview
- *_poster/* contains our poster pdf and its latex source.
- *_report/* contains our report pdf and its latex source.
- *_slides/* contains our weekly presentation pdfs.
- *code/* contains our source code. Move to this directory and follow the *README.md* inside to setup the environment and run the code.
  - *code/configs/* contain various config files for fine-tuning.