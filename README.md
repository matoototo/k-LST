# k-LST and SiVA

**Authors:** Baykam Say, Barış Coşlu, and Mato Gudelj \
**Supervisor:** M.Sc. Jeremias Bohn

This work introduces two novel Parameter-Efficient Fine Tuning (PEFT) methods for fine-tuning of large-scale pre-trained transformer models, Singular Value Adaptation (SiVA) and k-Ladder Side-Tuning (k-LST). Additionally, it utilizes input prompts to improve the performance of PEFT methods. SiVA improves upon LoRA by leveraging singular value decomposition for initialization, thereby avoiding the problematic zero-initialization employed by LoRA. This leads to faster convergence while maintaining full fine-tuning performance. k-LST improves upon Ladder Side Tuning, closing the performance gap of memory efficient PEFT methods while retaining a low memory footprint. It extracts backbone features with a sliding window, which are then queried by the side network with cross-attention. Prompts, when used in conjunction with various PEFT methods, improve performance over the baseline with no computational drawbacks.

## Repository Overview
- *_poster/* contains our poster pdf and its latex source.
- *_report/* contains our report pdf and its latex source.
- *_slides/* contains our weekly presentation pdfs.
- *code/* contains our source code. Move to this directory and follow the *README.md* inside to setup the environment and run the code.
  - *code/configs/* contain various config files for fine-tuning.
