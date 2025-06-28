# Onpremise LLM Normal Detection
Official codes for **Integrating ChatGPT into Secure Hospital Networks: A Case Study on Improving Radiology Report Analysis** on Conference on Health, Inference, and Learning (CHIL) 2024.

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

## Trained model ckpt
With the RadBERT baseline, sentence-level classification with the contrastive set up has great performance for the overall ablations (0.977 AUC).
On the following google links, you can download the trained [sentence level anomaly classifier](https://drive.google.com/file/d/1QuRSJBnaj5Plj_XAxRE8XsyjESLyS9wb/view?usp=drive_link).

## Model Performance

## Citation
    @InProceedings{kim2024sparse,
      title={},
      author={},
      journal={},
      year={2024}
    }
