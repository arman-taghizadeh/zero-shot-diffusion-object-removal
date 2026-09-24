# Zero-Shot Diffusion Object Removal

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arman-taghizadeh/zero-shot-diffusion-object-removal/blob/main/object_removal_demo.ipynb)

This repository contains the implementation accompanying the paper
**Zero-Shot Object Removal via Attention Masking, Latent Anchoring, and Refinement**
and the corresponding master's thesis.

The method removes selected objects from real images without task-specific training or model fine-tuning. It combines automatic image conditioning, inversion-based reconstruction, attention control, latent background preservation, and localized refinement within a pretrained Stable Diffusion model.

## Paper

**Zero-Shot Object Removal via Attention Masking, Latent Anchoring, and Refinement**  
Arman Taghizadeh, Ulf Krumnack, Kai-Uwe Kühnberger

[![arXiv](https://img.shields.io/badge/arXiv-2609.28342-b31b1b.svg)](https://doi.org/10.48550/arXiv.2609.28342)

The paper is available on arXiv:

https://doi.org/10.48550/arXiv.2609.28342


## Method Overview

The object-removal pipeline consists of the following stages:

1. **Object mask construction** using Segment Anything (SAM), followed by optional brush-based correction and mask postprocessing.
2. **Automatic image captioning** using BLIP to obtain the text condition used throughout the diffusion pipeline.
3. **DDIM inversion** to recover a latent trajectory corresponding to the input image.
4. **Background-weighted masked null-text optimization** to improve reconstruction of the preserved image region while leaving the target region sufficiently unconstrained for resynthesis.
5. **Decoder self-attention masking** to reduce propagation of target-object information during denoising.
6. **Hard latent anchoring** to restore the outside-mask region from the inversion trajectory at every denoising step.
7. **Localized renoise–denoise refinement** to reduce artifacts and boundary inconsistencies in the edited region.

The notebook additionally includes a **no-NTI ablation**, in which the same editing pipeline is executed using the fixed empty-prompt embedding instead of optimized unconditional embeddings.

## Notebook

The complete implementation is provided in:

```text
object_removal_demo.ipynb

The notebook is designed for execution in **Google Colab with a GPU runtime**.

## Requirements

The implementation was tested with:

```text
Python 3.13.15
PyTorch 2.11.0+cu128
torchvision 0.26.0+cu128
numpy 2.1.3
Pillow 11.3.0
OpenCV 4.14.0
matplotlib 3.11.1
transformers 5.15.1
diffusers 0.40.0
accelerate 1.14.0
safetensors 0.8.0
tqdm 4.67.3
LPIPS 0.1.4
```

Additional Python dependencies are listed in:

```text
requirements.txt
```

The implementation uses the following pretrained models:

- Stable Diffusion v1.4
- Segment Anything (SAM ViT-H)
- BLIP image captioning
- CLIP and pretrained ResNet models for evaluation

Model weights are not included in this repository and are downloaded through their respective libraries or official sources.

## Usage

Open the notebook using the **Open in Colab** button above.

Use a GPU runtime and execute the notebook sequentially. The notebook will:

1. install the required dependencies,
2. download the SAM checkpoint,
3. request an input image,
4. construct and optionally refine the object mask,
5. generate an automatic BLIP caption,
6. perform DDIM inversion,
7. optimize the masked null-text embeddings,
8. perform the primary object-removal process,
9. apply localized refinement,
10. evaluate the refined result, and
11. run the corresponding no-NTI ablation.

## Reproducibility

The main experimental parameters are collected in the notebook's **Configuration** section, including:

- number of diffusion steps,
- masked-NTI optimization parameters,
- classifier-free guidance scales,
- self-attention masking schedule,
- mask dilation,
- refinement parameters, and
- evaluation-region widths.

Random seeds are fixed in the notebook for reproducibility.

## Repository Structure

```text
.
├── object_removal_demo.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

## Citation

If you use this implementation in your research, please cite the associated paper:

**Arman Taghizadeh, Ulf Krumnack, and Kai-Uwe Kühnberger.**  
*Zero-Shot Object Removal via Attention Masking, Latent Anchoring, and Refinement.*  
arXiv:2609.28342, 2026.  
https://doi.org/10.48550/arXiv.2609.28342

Citation metadata is also provided through the repository's `CITATION.cff` file.

## Acknowledgements

This implementation builds on pretrained models and software including Stable Diffusion, Hugging Face Diffusers and Transformers, Segment Anything, BLIP, CLIP, ...

Please refer to the corresponding original projects and publications for their respective licenses and citation requirements.

## License

This repository is licensed under the
[PolyForm Noncommercial License 1.0.0](LICENSE).

The software may be used, modified, and distributed only for
noncommercial purposes as permitted by the license. Commercial use is
not permitted without separate permission from the copyright holder.

The pretrained models, model weights, and third-party libraries used by
this implementation remain subject to their respective licenses and
terms of use.
