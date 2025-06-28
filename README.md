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
We utilized the trained abnomaly sentence-level classificatio model, and you can downloaded the model on this link: [Anomaly-Classification](https://drive.google.com/file/d/1QuRSJBnaj5Plj_XAxRE8XsyjESLyS9wb/view?usp=drive_link).


## Model Performance

## Citation
    @InProceedings{kim2024sparse,
      title={},
      author={},
      journal={},
      year={2024}
    }
