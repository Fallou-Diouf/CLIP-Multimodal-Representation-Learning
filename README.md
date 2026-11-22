# Apprentissage de Représentations Multimodales Texte / Image via CLIP

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![OpenAI CLIP](https://img.shields.io/badge/Model-OpenAI%20CLIP-412991.svg)](https://github.com/openai/CLIP)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

Ce dépôt est réalisés dans le cadre du TP sur l'apprentissage de représentations bimodales **CLIP** (*Contrastive Language–Image Pre-training*) en Master 2 VMI 2026-2027 de l'Université Paris Cité.

---

## Objectifs et Structure du Projet

L'étude s'articule autour de quatre axes expérimentaux :

1. **Diagnostic de l'espace latent bimodal & *Domain Shift* (Consigne 1) :**
   * Inspection architecturale de `ViT-B/16` ($149.6\text{M}$ paramètres, encodeur visuel ViT + encodeur textuel Transformer BPE).
   * Analyse comparative de la similarité cosinus entre images naturelles (`scikit-image`) et imagerie radiologique (**ROCO**), mise en évidence du *domain shift* et évaluation de l'impact du *fine-tuning* contrastif.
   * Comportement d'un classifieur *Zero-Shot* à vocabulaire fermé (`CIFAR-100`) face à des images hors distribution.
2. **Benchmark architectural quantitatif (Consigne 2) :**
   * Confrontation de quatre architectures représentatives (`RN50`, `ViT-B/32`, `ViT-B/16`, `ViT-L/14`) sur un protocole équitable ($1\,000$ images de test `CIFAR-100`) : compromis précision (*Top-1 / Top-5 Accuracy*), latence d'inférence (ms/image) et complexité paramétrique.
3. **Évaluation médicale sans fuite de données sur ROCO (Consigne 3) :**
   * Évaluation quantitative de recherche cross-modale (*Image-to-Text* et *Text-to-Image Retrieval*) sur le split de validation indépendant `local-val-list.csv` ($N = 99$ paires radiologiques disjointes) avant et après *fine-tuning*.
4. **Généralisation *Zero-Shot* et *Prompt Engineering* sur `STL-10` (Consigne 4) :**
   * Évaluation hors distribution sur $1\,000$ images de test de `STL-10` ($96 \times 96$ px), étude d'ablation sur la formulation des prompts textuels et analyse fine de la matrice de confusion.

---

## Synthèse des Résultats Expérimentaux

### 1. Comparaison des architectures CLIP (`CIFAR-100`, $1\,000$ images de test)
| Modèle | Famille | Paramètres | Dim. Latent $d$ | Latence (ms/img) | Top-1 Acc. | Top-5 Acc. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`RN50`** | ResNet | $102.0\text{M}$ | $1024$ | $3.39\text{ ms}$ | $40.00\%$ | $71.80\%$ |
| **`ViT-B/32`** | ViT | $151.3\text{M}$ | $512$ | $3.36\text{ ms}$ | $61.80\%$ | $85.70\%$ |
| **`ViT-B/16`** | ViT | $149.6\text{M}$ | $512$ | $4.38\text{ ms}$ | $66.80\%$ | $88.00\%$ |
| **`ViT-L/14`** | ViT | $427.6\text{M}$ | $768$ | $13.02\text{ ms}$ | **$74.90\%$** | **$93.60\%$** |

### 2. Impact du *Fine-tuning* médical sur ROCO Validation ($N = 99$ paires disjointes)
| Configuration (`ViT-B/16`) | I2T R@1 | I2T R@5 | I2T R@10 | I2T MedR | T2I R@1 | T2I R@5 | T2I R@10 | T2I MedR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pré-entraîné (Brut)** | $22.22\%$ | $54.55\%$ | $73.74\%$ | $4.0$ | $22.22\%$ | $55.56\%$ | $68.69\%$ | $4.0$ |
| **Fine-tuné (ROCO)** | **$68.69\%$** | **$91.92\%$** | **$97.98\%$** | **$1.0$** | **$63.64\%$** | **$91.92\%$** | **$97.98\%$** | **$1.0$** |

### 3. Généralisation *Zero-Shot* et *Prompt Engineering* (`STL-10`, $1\,000$ images)
| Stratégie de Prompt | Top-1 Accuracy | Top-5 Accuracy | F1-Score Macro |
| :--- | :---: | :---: | :---: |
| **Label brut (`"{label}"`)** | $96.40\%$ | $99.90\%$ | — |
| **Prompt contextualisé (`"a photo of a {label}"`)** | **$97.60\%$** | **$100.0\%$** | **$0.976$** |

---

## Arborescence du Dépôt

```text
├── TP_CLIP_Multimodal_Representation_Learning.ipynb   # Notebook Jupyter complet (code + visualisations + analyses)
├── Rapport_CLIP_Multimodal.pdf                        # Compte rendu scientifique synthétique (3 pages)
└── README.md                                          # Documentation du projet
