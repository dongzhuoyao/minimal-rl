# minimal-rl

A minimal implementation and tutorial for FlowGRPO (Flow Matching with Group Relative Policy Optimization).

## Quick Links

- **Tutorial**: See `tutorial/` directory for a simplified tutorial implementation
- **Original Implementation**: See `original_impl/` for the full FlowGRPO codebase

## Tutorial

To learn the basics of FlowGRPO, check out the tutorial in the `tutorial/` directory. It includes:

- A simplified 1D flow matching model (easy to visualize)
- Toy dataset with simple prompts
- GRPO training implementation
- Visualization and evaluation tools

Quick start:
```bash
cd tutorial
pip install -r requirements.txt
python -m tutorial.dataset.generate_dataset
python tutorial/train.py
```

## Original Implementation

The `original_impl/` directory contains the full FlowGRPO implementation from the [original repository](https://github.com/yifan123/flow_grpo), which supports:

- SD3, FLUX, Qwen-Image, and other flow matching models
- Multiple reward functions (PickScore, OCR, GenEval, etc.)
- Multi-GPU and multi-node training
- FlowGRPO-Fast and GRPO-Guard variants

See `original_impl/README.md` for detailed documentation.