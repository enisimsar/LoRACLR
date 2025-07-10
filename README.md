<h1 align="center">LoRACLR: Contrastive Adaptation for Customization of Diffusion Models [CVPR 2025]</h1>

<p align="center">
    <a href="https://arxiv.org/abs/2412.09622">
    <img alt="Static Badge" src="https://img.shields.io/badge/arXiv-2412.09622-b31b1b"></a>
    <a href="https://loraclr.github.io/">
        <img alt="Static Badge" src="https://img.shields.io/badge/Page-Link-darkgreen"></a>
</p>

[Enis Simsar](https://enis.dev/), [Thomas Hofmann](https://da.inf.ethz.ch/), [Federico Tombari](https://federicotombari.github.io/), [Pinar Yanardag](https://pinguar.org/)

> Recent advances in text-to-image customization have enabled high-fidelity, context-rich generation of personalized images, allowing specific concepts to appear in a variety of scenarios. However, current methods struggle with combining multiple personalized models, often leading to attribute entanglement or requiring separate training to preserve concept distinctiveness. We present LoRACLR, a novel approach for multi-concept image generation that merges multiple LoRA models, each fine-tuned for a distinct concept, into a single, unified model without additional individual fine-tuning. LoRACLR uses a contrastive objective to align and merge the weight spaces of these models, ensuring compatibility while minimizing interference. By enforcing distinct yet cohesive representations for each concept, LoRACLR enables efficient, scalable model composition for high-quality, multi-concept image synthesis. Our results highlight the effectiveness of LoRACLR in accurately merging multiple concepts, advancing the capabilities of personalized image generation.

## Dependencies and Installation

- Python >= 3.10 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- Diffusers==0.19.3
- XFormer (is recommend to save memory). Run `pip install xformers`

## Single-Client Concept Tuning
- **Data:** Dataset should be prepared like described in [Mix-of-Show repository](https://github.com/TencentARC/Mix-of-Show?tab=readme-ov-file#data-preparation).
- **Training:** Please refer to training commands in [Mix-of-Show repositoty](https://github.com/TencentARC/Mix-of-Show?tab=readme-ov-file#computer-single-client-concept-tuning) and [Orthogonal Adaptation reposityory](https://huggingface.co/spaces/ujin-song/ortha/blob/main/README.md).

## Merging LoRAs
Some trained models can be obtained from [Orthogonal Adaptation reposityory](https://huggingface.co/spaces/ujin-song/ortha/tree/main/experiments), put the trained models in `experiments/` folder.

### Step 1: Collect Concept Models
Collect your concept models and update `config.json` accordingly.

```json
[
    {
        "lora_path": "experiments/single-concept/elsa/models/edlora_model-latest.pth",
        "unet_alpha": 1.5,
        "text_encoder_alpha": 1.5,
        "concept_name": "<elsa1> <elsa2>"
    },
    {
        "lora_path": "experiments/single-concept/moana/models/edlora_model-latest.pth",
        "unet_alpha": 1.5,
        "text_encoder_alpha": 1.5,
        "concept_name": "<moana1> <moana2>"
    }
    ... # keep adding new concepts for extending the pretrained models
]
```

### Step 2: Weight Fusion
Run the following command:

```bash
python weight_fusion.py \
    --concept_cfg config.json \
    --save_path ./experiments/multi-concepts \
    --pretrained_model nitrosocke/mo-di-diffusion
```

### Step 3: Sample
Use `inference.ipynb` notebook.

## Citation

If you find our work useful, please consider citing our paper:

```
@inproceedings{simsar2025loraclr,
  title={LoRACLR: Contrastive Adaptation for Customization of Diffusion Models},
  author={Simsar, Enis and Hofmann, Thomas and Tombari, Federico and Yanardag, Pinar},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={13189--13198},
  year={2025}
}
```

## Acknowledgment

This project builds upon the structure and pretrained weights from the [Ortha](https://huggingface.co/spaces/ujin-song/ortha) repository by [ujin-song](https://huggingface.co/ujin-song). We thank the authors for making their work publicly available.
