# Onpremise LLM Normal Detection

## Abstract
This study demonstrates the first in-hospital adaptation of a cloud-based AI, similar to ChatGPT, into
a secure model for analyzing radiology reports, prioritizing patient data privacy. By employing a unique
sentence-level knowledge distillation method through contrastive learning, we achieve over 95% accuracy in
detecting anomalies. The model also accurately flags uncertainties in its predictions, enhancing its reliability
and interpretability for physicians with certainty indicators. These advancements represent significant
progress in developing secure, efficient AI tools for healthcare, suggesting a promising future for in-hospital
AI applications with minimal supervision.

## Requirements
You can download the requirements using requirements.txt file.

<pre><code>$ conda create -n normal_detection
$ conda activate normal_detection
$ pip install -r requirements.txt
</code></pre>

## Trained model ckpt and test datasets
You can download the trained model ckpt and test dataset from following google drive link.

**Root**: https://drive.google.com/drive/folders/1QKoruoRKlEH7i1ABO8K0NHXOGaEdOjo2?usp=drive_link

**Sententance level KD + contrastive learning** : https://drive.google.com/file/d/14Hh1xOR5kYsqYwU-7db-4E7cCxtJbP6a/view?usp=drive_link

**Sentence level KD (without contrastive learning)**: https://drive.google.com/file/d/1oWh0nGzP6B3sprA_Su2dZJeO8E0Bjxpk/view?usp=drive_link

**Document level KD + contrastive learning**: https://drive.google.com/drive/folders/1scWk_9YwB4pqQJ--8zhLSmJuijA2DWnR?usp=drive_link

**Document level KD (without contrastive learning)**: https://drive.google.com/drive/folders/1IM4qH9U2FtQ9P8HbvOhmlOEHgkgEmcBA?usp=drive_link

**Test data**: https://drive.google.com/drive/folders/1uzyfIejR0nYAKvmXdinBceoLE04Z23WZ?usp=drive_link

## Model Description


## Model Performance

## Citation
    @InProceedings{kim2024sparse,
      title={},
      author={},
      journal={},
      year={2024}
    }
