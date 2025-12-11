# Sustainability Analysis: Adaptation of Attention U-Net for Satellite-Based Ocean Oil Spill Detection

This document provides a short report on the technical and sustainability aspects of adapting the Attention U-Net architecture for the detection of oil spills in satellite SAR imagery. It covers the replication of the baseline AI methodology, the contextual relevance of the marine oil spill detection challenge, scalability and sustainability considerations, dataset curation, model architecture adaptations, and evaluation results.

---

## 1. Replication of Baseline AI Methodology

### 1.1 Repository Cloning and Setup

**Original Repository:** The baseline Attention U-Net architecture was adapted from deforestation detection research, with the original implementation available in the `deforestation-adaptation/` directory.

### 1.2 Dependencies and Environment Setup

**Python Version:** 3.11+

| Package | Version | Purpose |
|---------|---------|---------|
| torch | ≥2.0.0 | Deep learning framework |
| torchvision | ≥0.15.0 | Image transformations |
| numpy | ≥1.24.0 | Numerical computing |
| Pillow | ≥9.0.0 | Image loading |
| scikit-learn | ≥1.2.0 | Metrics and data splitting |
| matplotlib | ≥3.7.0 | Visualization |
| wandb | ≥0.15.0 | Experiment tracking |
| tqdm | ≥4.65.0 | Progress bars |
| rasterio | ≥1.3.0 | GeoTIFF processing |

**Hardware for Baseline Reproduction:**
- GPU: NVIDIA RTX 5070
- RAM: 32GB system memory

### 1.3 Baseline Results Reproduction

See the notebook in `deforestation-adaptation/notebooks/deforestation_reproduction.ipynb` for replication of original results.

**Original Paper Metrics (Deforestation Detection):**

| Metric | Original Paper | Our Reproduction | Difference |
|--------|---------------|------------------|------------|
| Accuracy | 97.8% | 98.1% | +0.3% ✓ |
| IoU | 85.2% | 84.7% | -0.5% ✓ |
| F1-Score | 91.5% | 91.2% | -0.3% ✓ |
| Precision | 90.8% | 91.0% | +0.2% ✓ |
| Recall | 92.3% | 91.4% | -0.9% ✓ |

*All metrics within ±5% threshold ✓*

**Deforestation Reproduction Process:**
1. Download original Amazon deforestation dataset and run the following scripts:
    - preprocess-rgb-data.py
    - preprocess-4band-amazon-data.py
    - preprocess-4band-atlantic-forest-data.py
2. Clone Attention U-Net model and training scripts from `deforestation-adaptation/predictor.py`
2. Trained with same hyperparameters (50 epochs, Adam optimizer, lr=1e-4)
3. Evaluated on held-out test set

**Pre-trained Models Available:**
- `models/attention_unet_best.pth` - Baseline model

**HuggingFace Repository:** [huggingface/attention_u_net_oil_spill](https://huggingface.co/Bobsicle/attention_u_net_oil_spill)

---

## 2. Contextually Relevant Challenge: Marine Oil Spill Detection

### 2.1 Problem Definition and SDG Alignment

1. Ocean life is critical to humanity, based on UN’s Food and Agriculture Organization report: 
    - Supports the livelihood of more than 3 billion people
    - Produces 50% of earth’s O2
    - Absorbs 30% of earth’s CO2

2. Oil spills severely damage the ecosystem by killing marine life directly, as well as poison their habitats for years, potentially leading to a change in the food chain.

3. The monitoring of oceans is very difficult given that it encompasses two-thirds of the earth’s surface. With most ocean territory being international waters, it is also extremely hard to enforce laws over oil spills.

These challenges point to the need for automated ocean monitoring, and Attention U-Net provides a strong solution by being able to focus on subtle patterns while suppressing the many irrelevant background noise of the ocean.


#### **Primary SDG Alignment:**

#### SDG 14: Life Below Water

*"Conserve and sustainably use the oceans, seas and marine resources for sustainable development"*

| Target | Project Contribution |
|--------|---------------------|
| **14.1** - Prevent and significantly reduce marine pollution | Reducing marine pollution by early oil spill detection. |
| **14.2** - Sustainably manage and protect marine ecosystems | Supports protection of marine ecosystem by monitoring.|
| **14.a** - Increase scientific knowledge for ocean health | Contributes to enhancing ocean science. |

#### SDG 13: Climate Action

*"Take urgent action to combat climate change and its impacts"*

| Target | Project Contribution |
|--------|---------------------|
| **13.1** - Strengthen resilience to climate-related hazards | Increasing resilience and adaptive capacity to climate-related hazards. |
|**13.2** - Integrate climate change measures into policies and planning | Provides data to support further climate action or policies.|
| **13.3** - Improve education and awareness on climate | Provides foundation for autonomous monitoring and awareness of oil spills. |

#### SDG 12: Responsible Consumption and Production

*"Ensure sustainable consumption and production patterns"*

| Target | Project Contribution |
|--------|---------------------|
| **12.4** - Environmentally sound management of chemicals and wastes | Supports safe management of toxics and chemicals by autonomous monitoring and early detection. |
| **12.6** - Encourage Companies to Adopt Sustainable Practices and Sustainability Reporting |Encourages companies to adopt sustainable practice by raising accountability.|

#### **Possible Negative SDG Interactions:**

| Target | Description | Mitigation |
|--------|-------------|------------|
| **SDG 7.2 — Increased energy usage for model training** | Training and large-scale inference raise electricity demand and emissions. | Efficient training (mixed precision, smaller models), model distillation, carbon-aware scheduling, and using low‑carbon data centers. |
| **SDG 8.5 — Automation may reduce certain job roles** | Automated monitoring could displace manual inspection and related operational roles. | Retraining and upskilling, redeployment to higher‑value tasks, stakeholder engagement, and phased implementation. |
| **SDG 9.1 — Reliance on advanced compute and satellite systems may widen technological inequalities** | High-cost infrastructure and expertise favor well‑resourced actors and regions. | Open‑source tools, low‑cost inference options (edge/cloud hybrid), capacity building, partnerships, and documentation. |

### 2.2 Limitations and Ethical Considerations

#### Technical Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Lookalike Phenomena** | Natural slicks, algal blooms cause false positives | Provide more data for classification (Oil, Lookalike, No oil) |
| **Weather Effects** | Extreme winds disperse oil signatures | Temporal analysis, wind speed thresholds |
| **Model Generalization** | Performance may vary across regions | Fine-tuning with regional data recommended |

#### Ethical Considerations

| Consideration | Analysis | Mitigation |
|---------------|----------|------------|
| **False Positives** | Wasted response resources, unnecessary alarms, degradation in trust | Involvement of some human verification |
| **Data Privacy** | Ship tracking correlation with spills | Anonymization of vessel data |
| **Environmental Justice** | Developing nations may lack implementation capacity | Open-source release, documentation, capacity building |

### 2.3 Scalability and Sustainability Analysis

#### Scalability Assessment

| Dimension | Current State | Scalability Path |
|-----------|--------------|------------------|
| **Geographic** | European Sea | Sentinel-1 data available worldwide, more labelled annotations are required however for better performance |
| **Computational** | Single GPU | Increased compute resources through AWS or other cloud integration |

#### Long-term Sustainability

- **Hardware Lifecycle** Sentinel-1 operational until 2030+ with successor missions planned, ensuring data continuity for long term
- **Model Maintenance:** Transfer learning enables adaptation to new sensors
- **Community Support:** Open-source enables distributed maintenance

---


## 3. Dataset Curation and Preprocessing 

### Covered in depth in [README_DATA_GUIDE.md](./README_DATA_GUIDE.md)

**Selected Dataset:** Sentinel-1 SAR Oil Spill Detection Dataset from previous work by Trujillo-Acatitla et al. (2024)

| Attribute | Specification |
|-----------|--------------|
| **Source** | European Space Agency (ESA) Copernicus Programme |
| **Satellite** | Sentinel-1A/B |
| **Sensor** | C-band SAR |
| **Mode** | Interferometric Wide Swath (IW) |
| **Resolution** | 10m × 10m |
| **Polarization** | VV/VH |
| **Geographic Coverage** | Global oceans |
| **Temporal Coverage** | 2014-present |

---

**Ethical Considerations of Sentinel-1 Dataset:**

There are no direct ethical concerns with the Sentinel-1 dataset itself, as it contains no personal data. However, the following considerations are noted:

| Aspect | Consideration |
|--------|---------------|
| **Data Licensing** | Copernicus data free and open under CC BY 4.0 |
| **Attribution** | "Contains modified Copernicus Sentinel data [2024]" |
| **Privacy** | No personal data in satellite imagery |
| **Consent** | Public domain environmental monitoring |

---

## 4. Model Architecture Adaptation

### 4.1 Architectural Modifications

**Baseline Architecture (Deforestation):**
- Input: 4-channel amazon rainforest imagery (RGB + NIR) from Sentinel-2
- Task: Binary segmentation (forest/deforested)
- Context: Amazon rainforest

**Adapted Architecture (Oil Spill):**
- Input: 1-channel SAR backscatter imagery from Sentinel-1 (VV polarization)
- Task: Binary segmentation (oil/background)
- Context: Ocean surface

---

**Modifications Made:** Model has been adapted with a **weighted binary cross-entropy** loss function to address class imbalance, as oil pixels are sparse compared to the ocean background.

### 4.2 Hyperparameter Adjustments

| Hyperparameter | Original Value (Deforestation) | Adapted Value (Oil Spill) | Rationale |
|----------------|-------------------------------|---------------------------|-----------|
| Loss Function | BCE | Weighted BCE | Address class imbalance |
| Learning Rate | 5e-4 | 1e-4 | In order to improve stability |
| Batch Size | 4 | 8 | Increased for better GPU utilization |
| Epochs | 50 | 20 | Increased runtime due to datasize |


---

## 5. Model Evaluation and Comparison

### Performance Comparison: Original vs. Adapted

**Reproduced Baseline (Deforestation Detection):**

| Baseline vs Reproduced | Accuracy | Precision | Recall | F1 |
|------------------------|----------|-----------|--------|-----|
| Baseline (Paper) | 97.5% | 97.6% | 97.5% | 97.5% |
| Reproduced | 97% | 98.1% | 95.9% | 97% |

**Adapted Model (Oil Spill Detection):**
| Metrics | BCE | Weighted BCE (Recommended) | Δ from Reproduced Baseline |
|---------|-----|---------------------------|-----------------|
| Accuracy | 97.9% | **98.3%** | +1.3% |
| Precision | 87.6% | 74.1% | -24.0% |
| Recall | 29.1% | **58.8%** | -37.1% |
| F1-Score | 43.7% | **65.6%** | -31.4% |
| IoU | 28.0% | **48.8%** | - |

**Performance Analysis:** 

Model achieved high accuracy, but scores lower on other metrics.
- Accuracy is high as “no oil” cases are abundant.
- Model can detect oil spills correctly but does not segment spills perfectly.

The adapted model shows lower performance than the baseline due to:

1. **Task Complexity**: Oil spills have irregular, diffuse boundaries vs. clear forest edges
2. **Class Imbalance**: Oil pixels are much sparse (<5% of ocean images and pixels)
3. **Lookalike Confusion**: Natural slicks mimic oil signatures
4. **Epoch Limitations**: Fewer training epochs (20 vs 50) due to compute constraints

### Metrics Appropriateness

| Metric | Appropriateness | Rationale |
|--------|-----------------|-----------|
| **Accuracy** | Not Appropriate | Class imbalance makes accuracy misleading, this is only kept for comparison |
| **Precision** | ✓ Important | Minimizes false alarm costs |
| **Recall** | ✓ Critical | Missed spills have severe consequences |
| **F1-Score** | ✓ Primary | Balances precision and recall |
| **IoU** | ✓ Primary | Measures segmentation quality |