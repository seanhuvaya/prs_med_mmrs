# PRS-Med (reproduction)

Implements the PRS-Med pipeline:
- Tiny vision encoder → token-level MLLM embeddings → cross-attention fusion → upsampling mask
- Joint loss: BCE+Dice (seg) + CE (text)  # per Eq.(5–7)   <!-- cites paper -->
- LoRA adapters on the MLLM backbone  <!-- cites paper -->

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
