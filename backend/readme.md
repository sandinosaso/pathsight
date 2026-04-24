# 🔬 Model Benchmark Study: Transfer Learning

This document outlines the investigation into pre-trained CNN architectures for binary classification of the **PatchCamelyon (PCam)** dataset.

## 📊 Comparison Table

| Architecture | Parameters | Top-1 Acc (ImageNet) | Optimal Resolution | Preprocessing Style | Key Advantage |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EfficientNetB0** | 5.3M | 77.1% | $224 \times 224$ | Internal $[0, 255]$ | Best Accuracy/Parameter ratio. |
| **MobileNetV2** | 3.4M | 71.3% | $224 \times 224$ | Scaled $[-1, 1]$ | Lowest latency for edge deployment. |
| **ResNet50** | 25.6M | 76.0% | $224 \times 224$ | Caffe (Mean Subtraction) | Residual links prevent vanishing gradients. |
| **ConvNeXt Tiny** | 28.6M | 82.1% | $224 \times 224$ | ImageNet Norm | Modern "pure CNN" with SOTA performance. |

---

## ⚙️ Requirements & Constraints

### 1. Image Size Optimization
Most models achieve peak feature extraction at **224x224**. Since our native PCam data is **96x96**, we are evaluating two strategies:
* **Upscaling:** Bicubic upscaling to 224 allows for better feature extraction by pre-trained ImageNet layers.
* **Input Adaptation:** Setting `input_shape=(96, 96, 3)` and relying on Global Average Pooling to handle smaller feature maps.

### 2. Preprocessing Sensitivity
The models are highly sensitive to their specific preprocessing pipelines.

* **EfficientNet:** Contains a built-in `Rescaling` layer; expects raw pixel values.
* **ConvNeXt:** Requires specific ImageNet normalization (Mean/Std dev subtraction).
* **ResNet/MobileNet:** Requires scaling to a $[-1, 1]$ range.

---

## 🧪 Experimental Roadmap
1.  **Baseline:** Train a classification head (GAP + Dense 128 + Dropout + Dense 1) on a frozen **EfficientNetB0**.
2.  **Cross-Validation:** Compare Validation AUC and Top-1 Accuracy across all four backbones.
3.  **Fine-Tuning:** Unfreeze the top 20 layers of the best-performing model to adapt specifically to histopathology textures.
