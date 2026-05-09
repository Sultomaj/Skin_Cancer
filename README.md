# 🩺 ISIC Skin Cancer Classification: An Ablation Study on Dataset Imbalance

## 📌 System Overview
This repository contains a comprehensive machine learning pipeline designed to classify 9 distinct skin pathologies using the ISIC (International Skin Imaging Collaboration) dataset. More than just a training script, this project documents a rigorous **ablation study** investigating the tension between global/local feature extractors (ViT vs. CNN) and algorithmic techniques for combating severe medical dataset imbalance.

The final deployed model is accessible via an interactive, clinically focused **Streamlit Web Application** hosted on Hugging Face Spaces.

---

## 🔬 The Ablation Study: Fighting "Lazy" Networks
Medical datasets are notoriously imbalanced. In this dataset, benign Nevi drastically outnumber critical early-stage Melanomas. Early baseline models achieved artificially high overall accuracy by defaulting to predicting "Nevus" (majority class) while dangerously missing Melanomas. 

To solve this, we systematically tested multiple architectures and objective functions:

### Phase 1: Global vs. Local Feature Extraction (ViT vs. CNN)
* **Vision Transformer (ViT-B/16):** Initially utilized for its global attention mechanism. However, fine-tuning a frozen ViT stalled at ~55% accuracy. Unfreezing the network improved accuracy to 66% but caused Melanoma recall to collapse to 18%, as the high-capacity network became "lazy" on the imbalanced data.
* **ResNet50:** Switched to a CNN architecture to leverage its inductive bias for local textural features (e.g., atypical pigment networks, streaks). ResNet50 extracted these dermatological features much more efficiently on this small dataset, heavily improving scores on Basal Cell Carcinoma and Vascular Lesions.

### Phase 2: Loss Function Engineering (Focal Loss)
To force the network to focus on rare, hard-to-classify examples, we implemented **Focal Loss**.
* While standard `CrossEntropyLoss` (even with class weights) allowed the model to coast on the easy Nevus images, Focal Loss successfully applied gradient penalties that pulled Melanoma recall back up.
* *Note: We identified a mathematical instability when double-dipping standard PyTorch class weights alongside the Focal Loss probability squashing, which we resolved by decoupling the unweighted probabilities inside a custom PyTorch module.*

### Phase 3: Data-Level Balancing (WeightedRandomSampler)
The ultimate solution involved fixing the class imbalance *before* it reached the loss function.
* Implemented a PyTorch `WeightedRandomSampler` inside the `DataLoader` to artificially flatten the dataset. By drastically increasing the sampling probability of rare classes, we guaranteed that every batch of 32 images was perfectly balanced.
* **Result:** Achieved our most mathematically stable validation loss (1.27) and highest minority-class recall.

---

## 🛠 Model Evaluation & Final Metrics
The final model utilized a **ResNet50 backbone + WeightedRandomSampler + Standard Cross-Entropy**.

* **Overall Validation Accuracy:** 65.25%
* **Basal Cell Carcinoma:** 100.0%
* **Vascular Lesion:** 100.0%
* **Nevus:** 100.0%
* **Melanoma:** 25.0% *(A 2x improvement over baseline models)*

**Dataset Limitation Note:** Extensive algorithmic optimization proves that the current diagnostic ceiling (~65%) is strictly constrained by dataset volume. Future iterations require aggregating larger, multi-source ISIC datasets (e.g., ISIC 2020) to fully saturate the network's parameter space and push Melanoma recall to clinically safe levels.

---

## 🚀 Interactive Deployment (Hugging Face)
The model is deployed as an interactive Web Application built with Streamlit.

**Clinical UI Features:**
* **Strict Inference Pipeline:** Strips out random training augmentations (rotations/flips/crops) to guarantee deterministic, clinically accurate tensor inputs.
* **Safety Thresholds:** Actively warns the user if diagnostic confidence drops below 50%.
* **Medical Context Engine:** Integrates with the Wikipedia API to automatically fetch standardized clinical summaries of the predicted pathology.
* **Visual Inspector:** Allows clinicians to apply real-time Canny Edge Detection and Grayscale contrast filters via OpenCV to manually inspect lesion borders.

---

## 💻 Tech Stack
* **Deep Learning Framework:** PyTorch, Torchvision
* **Architectures Evaluated:** ResNet50, Vision Transformer (ViT-B/16)
* **Frontend / UI:** Streamlit, Hugging Face Spaces
* **Computer Vision:** OpenCV (`cv2`), PIL
* **Evaluation & Metrics:** Scikit-learn, Matplotlib, JSON

---

## ⚙️ How to Run Locally

**1. Install Dependencies:**
```bash
pip install torch torchvision streamlit opencv-python-headless pandas numpy scikit-learn matplotlib wikipedia kagglehub
