# Getting Started with PerceptionMetrics

PerceptionMetrics is a comprehensive tool for evaluating and benchmarking object detection models on various datasets.

## Prerequisites

- Python 3.8+
- pip or conda

## Installation

```bash
pip install perceptionmetrics
```

## Quick Start

### 1. Load a Dataset

```python
from perceptionmetrics.datasets import CocoDataset

dataset = CocoDataset(
    annotations_file="path/to/annotations.json",
    images_dir="path/to/images"
)
```

### 2. Evaluate a Model

```python
from perceptionmetrics.models import YOLOv5Model

model = YOLOv5Model(model_size="small")
results = model.evaluate(dataset)
```

### 3. View Metrics

```python
print(f"mAP: {results.map}")
print(f"Precision: {results.precision}")
print(f"Recall: {results.recall}")
```

## Examples

### Training and Evaluation Example
See `../examples/tutorial_image_detection.ipynb` for a complete example.

### YOLO-specific Example
Run `../examples/tutorial_image_detection_yolo.ipynb` for YOLO model evaluation.

## Common Tasks

### Comparing Multiple Models

```python
from perceptionmetrics.evaluator import Evaluator

evaluator = Evaluator(dataset=dataset)

models = [model1, model2, model3]
comparison = evaluator.compare_models(models)
comparison.plot_results()
```

## Troubleshooting

- **Import errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
- **Dataset issues**: Check that file paths are correct and datasets are properly formatted
- **Memory errors**: Reduce batch size or dataset size

## Next Steps

- Read the [full documentation](../README.md)
- Check out advanced examples in the `examples/` directory
- Join our community on GitHub

## Contributing

We welcome contributions! See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.
