# MNIST Handwritten Digit Classifier

A neural network that recognizes handwritten digits (0–9) with ~96% accuracy,
built from scratch in TensorFlow/Keras.

## Dataset
MNIST — 70,000 images of handwritten digits (60,000 train / 10,000 test),
each 28×28 grayscale pixels. Built into TensorFlow via
`tf.keras.datasets.mnist`.

## Approach
- Flattened each 28×28 image into a 784-length vector; scaled pixel values to 0–1
- Neural network:
  - Input: 784 features (pixels)
  - Hidden: Dense(25, ReLU) → Dense(15, ReLU)
  - Output: Dense(10, linear), using SparseCategoricalCrossentropy(from_logits=True)
    for numerical stability
- Optimizer: Adam (learning rate 0.001), trained for 10 epochs
- Predictions via argmax over the 10 output logits; softmax applied to read
  confidence

## Results
- **Test accuracy: 95.85%** (on 10,000 unseen images)
- Confusion-matrix analysis shows the model's errors cluster among visually
  similar digits — 4↔9, 3↔5, 2↔8, 7↔9 — the same pairs a human might confuse
  in messy handwriting. The digit "1" is recognized most reliably.

## What I learned / would improve
- A dense network flattens the image and discards its 2D spatial structure.
  A convolutional network (CNN) would preserve that structure and typically
  push accuracy above 99%.
- Next steps: tune the architecture, add regularization, and experiment with
  a CNN.

## Files
- `mnist_classifier.ipynb` — full implementation

## Key concepts practiced
Multiclass classification · softmax · ReLU · Adam optimizer ·
from_logits numerical stability · confusion-matrix evaluation
