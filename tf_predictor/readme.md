# Bridging the Sim-to-Real Gap in Table Tennis Robots with a Transformer-Based Ball State Predictor

## Project Page

Visit our project page:

*Project page link coming soon.*

---

## Abstract

Robotic table tennis is a representative benchmark for high-speed, closed-loop robotic control in dynamic environments, where accurate and timely prediction of ball states is critical for reliable planning and control. Physics-based approaches rely heavily on accurate parameter identification and precise estimation of the initial ball state, while learning-based methods often struggle to capture long-range temporal dependencies and are typically trained on limited or simulated datasets.

We propose a transformer-based framework for table tennis ball state prediction that leverages attention mechanisms to model long-range temporal correlations directly from historical observations, without relying on explicit flight or bounce models. To support robust learning and strong generalization, we collected a large-scale real-world dataset from players of varying skill levels and diverse ball cannon configurations. The combination of a high-capacity transformer architecture and extensive real-world data enables accurate long-horizon prediction.

Building on this capability, we introduce a plug-and-play sim-to-real transfer strategy, **Swap Predictor at Deployment (SPAD)**, which replaces the physics-based predictor used during policy training with the proposed real-world-trained predictor at deployment. This improves the sim-to-real transferability of the learned policy without requiring any retraining. Experimental results demonstrate that this simple substitution effectively narrows the sim-to-real gap while preserving the efficiency and scalability of simulation-based training.

---

## Narrated Video

To learn more about our work, watch the narrated video:

https://drive.google.com/file/d/1Ak6SLjyQZd7JERYTWk_7P6PY8Cyp-ygT/view?usp=sharing

---

## Supplementary Material

Additional experiments and analyses that could not be included in the paper are provided in **`Supplementary_Material.pdf`**.

---

## Dataset

The test dataset is available at:

https://drive.google.com/file/d/10arpwhlQxeDZ1qyL9y0kN4QNYdvtOAhA/view?usp=sharing

The following example demonstrates how to load a sample dataset:

```python
import numpy as np

ball_data = np.load("sample1.npz")

ball_pos = ball_data["pos"]        # Ball positions (m)
ball_vel = ball_data["vel"]        # Ball velocities (m/s)
ball_spin = ball_data["spin"]      # Ball spin (rad/s)
ball_timestamps = ball_data["ts"]  # Timestamps (s)
```

Each `.npz` file contains the following arrays:

| Key | Description | Unit |
|------|-------------|------|
| `pos` | Ball positions | m |
| `vel` | Ball velocities | m/s |
| `spin` | Ball spin | rad/s |
| `ts` | Timestamps | s |

---

## Citation

If you find this work useful in your research, please consider citing our paper.

For details on the model architecture, training procedure, and evaluation, please refer to our paper (link will be updated once the preprint is available).

```bibtex
@misc{Bi2026,
  title={Bridging the Sim-to-Real Gap in Table Tennis Robots with a Transformer-Based Ball State Predictor},
  author={Yin Bi and Christian Conti and Bilan Yang and Alexander Sigrist and Peter Dürr and Naoya Takahashi},
  year={2026},
  eprint={},
  archivePrefix={arXiv},
  primaryClass={cs.SD},
  url={https://arxiv.org/abs/}
}
```
