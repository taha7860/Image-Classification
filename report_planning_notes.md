# Report Planning Notes

Do not paste these notes as the report. Use them to structure your own writing and connect your actual results to the marking criteria.

## Suggested Page Budget

- Introduction and robotics context: 0.5 page.
- Literature review: 1.25 pages.
- Methodology: 1 page.
- Results and interpretation: 1.75 pages.
- Conclusion and future work: 0.5 page.

## Literature Review Angle

Focus on deep learning for robot visual perception and object recognition:

- CNNs learn hierarchical visual features and are a strong basis for object recognition.
- Robot vision differs from static web-image classification because robots face viewpoint changes, lighting variation, clutter, limited training examples, and compute constraints.
- RGB-D and robot-captured datasets address embodiment and viewpoint variation better than CIFAR-style datasets.
- Transformers and transfer learning are useful current directions, but small CNN experiments remain useful for showing how architecture and hyperparameters affect generalisation.

## Methodology Points To Cover

- CIFAR-100 has 100 fine-grained classes, making it harder than CIFAR-10 and more appropriate for testing model capacity and regularisation.
- Use a validation split from the training data for model selection.
- Keep the test set untouched until final evaluation.
- Compare the same model under controlled hyperparameter changes.
- Explain why each hyperparameter matters:
  - Learning rate affects convergence speed and stability.
  - Dropout affects overfitting.
  - Batch size affects optimisation noise and training stability.
  - Augmentation tests robustness to visual variation.

## Results To Interpret

- Compare MLP versus CNN if both are run.
- Discuss train-validation gaps as evidence of overfitting or better generalisation.
- Use the confusion matrix or per-class accuracy to identify categories the model struggles with.
- Link augmentation effects back to robotics: viewpoint and lighting variation are normal in robot perception.

## Future Work

- Repeat the experiment on RGB-D Object Dataset or iCubWorld.
- Add depth channels or multimodal RGB-D fusion.
- Use transfer learning from an ImageNet-pretrained CNN.
- Evaluate robustness to viewpoint, occlusion, lighting, and object-instance splits.
