## Proactive Defense Against Deep Facial Attribute Editing via Non-Targeted Adversarial Perturbation Attack in the DWT domain

![Python 3.](https://img.shields.io/badge/python-3.11-green.svg?style=plastic)

![PyTorch 1.10.0](https://img.shields.io/badge/pytorch-2.1.2-green.svg?style=plastic)

> **Abstract:** *Proactive defense against face forgery disrupts the generative ability of forgery models by adding imperceptible perturbations to the faces to be protected. The recent latent space algorithm, i.e., Latent Adversarial Exploration (LAE), achieves better perturbation imperceptibility than the commonly-used image space algorithms but has two main drawbacks: (a) the forgery models can successfully edit the nullifying outputs after defense again; (b) the semantic information of defensed images is prone to be altered. Therefore, this paper proposes a proactive defense algorithm against deep facial attribute editing via non-targeted adversarial perturbation attack in the DWT domain. To address the former drawback, the nullifying attack is replaced by the non-targeted attack. Regarding the latter one, the perturbations are performed in the DWT domain. Furthermore, to speed user-concerned inference time, the generator-based approach is considered for generating frequency domain perturbations instead of the iterative approaches; to improve the visual quality of the defensed images, the perturbations are added in chrominance channels of YCbCr color space because the Human Visual System (HVS) is more sensitive to the perturbations in luminance channel. Numerous experimental results indicate that the proposed algorithm outperforms some existing algorithms, effectively disrupting the facial forgery system while achieving perturbation imperceptibility.*

## Pre-trained Models

Please download the pre-trained models from the following links and save them to `checkpoints/`

| SM                                                           | SA                                                           | PG                                                           |
| :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| Pretrained [FGAN](https://drive.google.com/file/d/1PQ5yZZ3lnyfN_gtShcdHdcDjCoHmwLXd/view?usp=sharing) Model. | [Saliency Detection Model](https://drive.google.com/file/d/1nwVloVzRLOGs7QL8QbBK0HA8ur9wPCTg/view?usp=sharing) | [Perturbation Generator](https://drive.google.com/file/d/17Lwzd_0NMW8_uE3ofJ53vC_d6-5Ac_9k/view?usp=sharing) |

## Get service online

Enable this flask app on port 5090:

```shell
pipenv install
pipenv run python -m flask app.py --port 5090
```

## Acknowledgment

This repo is based on [Fixed-Point-GAN](https://github.com/mahfuzmohammad/Fixed-Point-GAN) 、 [Adversarial-visual-reconstruction](https://github.com/NiCE-X/Adversarial-visual-reconstruction) and [TAFIM](https://github.com/shivangi-aneja/TAFIM), thanks for their great work.

