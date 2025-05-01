## LaForge10k: Synthetic Herbarium Specimen Images with Semantic Segmentation Annotations

This repository contains the code used to evaluate the influence of synthetic data on semantic segmentation tasks in herbarium specimen images.

### Synthetic Dataset

**LaForge10k** ([Zenodo link](https://doi.org/10.5281/zenodo.14675918)) is a procedurally generated synthetic training dataset for semantic segmentation of herbarium specimen images.

#### Class IDs in LaForge10k

Class IDs are intentionally spaced to be visually distinct:

```
class_ids = {'other': 0,  
             'paper': 5,  
             'crc': 50,  
             'scale': 100,  
             'woody': 150,  
             'label': 200,  
             'leaf': 250}
```
---

## /training_code

- `run_experiments_mixing.py` — Evaluates segmentation model performance using various combinations of real and synthetic training data.  
- `Milleville_mixing_config.yaml` — Example configuration file for an experimental suite.

---

## /dataset_conversions

- `Hussein_dataset_conversion.ipynb` — Notebook for converting the Hussein dataset to a harmonized format.  
- `Milleville_dataset_conversion.ipynb` — Notebook for converting the Milleville dataset to a harmonized format.

---

## Validation Datasets

This work would not be possible without the real-world validation datasets. We thank the authors for making their segmentation training data publicly available.

- **Hussein et al. (2020):**  
  Hussein, B. R., Malik, O. A., Ong, W. H., & Slik, J. W. F. (2020). *Semantic segmentation of herbarium specimens using deep learning techniques*. In *Computational Science and Technology: 6th ICCST 2019, Kota Kinabalu, Malaysia, 29–30 August 2019* (pp. 321–330). Springer Singapore.

- **Milleville et al. (2023):**  
  Milleville, K., Chandrasekar, K. K. T., Van de Weghe, N., & Verstockt, S. (2023, October). *Evaluating segmentation approaches on digitized herbarium specimens*. In *International Symposium on Visual Computing* (pp. 65–78). Cham: Springer Nature Switzerland.
