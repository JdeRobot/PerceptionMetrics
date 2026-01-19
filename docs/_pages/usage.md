---
layout: home
title: Usage
permalink: /usage/

sidebar:
  nav: "main"
---

## Interactive GUI
PerceptionMetrics now includes a **Streamlit-based GUI** that provides an intuitive interface for image detection model evaluation and dataset exploration.

### Launching the GUI
```bash
# From the project root directory
streamlit run app.py
```

### GUI Features
The GUI consists of three main tabs:

#### Dataset Viewer
- Browse and visualize your datasets
- View images and annotations
- Navigate through different splits (train/val)

#### Inference
- Run real-time inference on individual images
- Upload custom images for testing
- Visualize detection results with bounding boxes
- Adjust model parameters interactively

#### Evaluator
- Perform comprehensive model evaluation
- Real-time progress tracking by configuring evaluation step parameter
- Download results

## Library
🧑‍🏫️ [Image Segmentation Tutorial](https://github.com/JdeRobot/PerceptionMetrics/blob/master/examples/tutorial_image_segmentation.ipynb)

🧑‍🏫️ [Image Detection Tutorial](https://github.com/JdeRobot/PerceptionMetrics/blob/master/examples/tutorial_image_detection.ipynb)

You can check the [`examples` directory](https://github.com/JdeRobot/PerceptionMetrics/tree/master/examples) for inspiration. If you are using *poetry*, you can run the scripts provided either by activating the created environment using `poetry shell` or directly running `poetry run python examples/<some_python_script.py>`.

#### [Full docs for the Python library](https://jderobot.github.io/PerceptionMetrics/py_docs/build/html/index.html)

## Command-line interface
PerceptionMetrics currently provides a CLI with two commands, `pm_evaluate` and `pm_batch`. Thanks to the configuration in the `pyproject.toml` file, we can simply run `poetry install` from the root directory and use them without explicitly invoking the Python files.

#### `pm_evaluate`
Run a single evaluation job given a model and dataset configurations.

**Segmentation Example:**
```shell
pm_evaluate segmentation image --model_format torch --model /path/to/model.pt --model_ontology /path/to/ontology.json --model_cfg /path/to/cfg.json --dataset_format rellis3d --dataset_dir /path/to/dataset  --dataset_ontology /path/to/ontology.json --out_fname /path/to/results.csv
```

**Detection Example:**
```shell
pm_evaluate detection image --model_format torch --model /path/to/model.pt --model_ontology /path/to/ontology.json --model_cfg /path/to/cfg.json --dataset_format coco --dataset_dir /path/to/coco/dataset --out_fname /path/to/results.csv
```

Docs:
```shell
Usage: pm_evaluate [OPTIONS] {segmentation|detection} {image|lidar}

  Evaluate model on dataset

Options:
  --model_format [torch|tensorflow]
                                  Trained model format  [default: torch]
  --model PATH                    Trained model filename (TorchScript) or
                                  directory (TensorFlow SavedModel)
                                  [required]
  --model_ontology FILE           JSON file containing model output ontology
                                  [required]
  --model_cfg FILE                JSON file with model configuration (norm.
                                  parameters, image size, etc.)  [required]
  --dataset_format [gaia|rellis3d|goose|generic|rugd|coco]
                                  Dataset format  [default: gaia]
  --dataset_fname FILE            Parquet dataset file
  --dataset_dir DIRECTORY         Dataset directory (used for 'Rellis3D',
                                  'Wildscenes', and 'COCO' formats)
  --split_dir DIRECTORY           Directory containing .lst or .csv split
                                  files (used for 'Rellis3D' and 'Wildscenes'
                                  formats, respectively)
  --train_dataset_dir DIRECTORY   Train dataset directory (used for 'GOOSE'
                                  and 'Generic' formats)
  --val_dataset_dir DIRECTORY     Validation dataset directory (used for
                                  'GOOSE' and 'Generic' formats)
  --test_dataset_dir DIRECTORY    Test dataset directory (used for 'GOOSE' and
                                  'Generic' formats)
  --images_dir TEXT               Directory containing data (used for 'RUGD'
                                  format)
  --labels_dir TEXT               Directory containing annotations (used for
                                  'RUGD' format)
  --data_suffix TEXT              Data suffix to be used to filter data (used
                                  for 'Generic' format)
  --label_suffix TEXT             Label suffix to be used to filter labels
                                  (used for 'Generic' format)
  --dataset_ontology FILE         JSON containing dataset ontology (used for
                                  'Generic' and 'Rellis3D' formats)
  --split TEXT                    Name of the split or splits separated by
                                  commas to be evaluated  [default: test]
  --ontology_translation FILE     JSON file containing translation between
                                  dataset and model classes
  --out_fname PATH                CSV file where the evaluation results will
                                  be stored
  --predictions_outdir PATH       Directory where predictions (images/points
                                  and CSV) per sample will be stored. If not
                                  provided, predictions per sample will not be
                                  saved
  --help                          Show this message and exit.
```

#### `pm_batch`
Execute requested jobs sequentially. It must be configured by means of a YAML file.

Example:
```shell
pm_batch evaluate /path/to/batch_config.yaml
```

Docs:
```shell
Usage: pm_batch [OPTIONS] {evaluate|computational_cost} JOBS_CFG

  Perform detection metrics jobs in batch mode

Options:
  --help  Show this message and exit.
```

YAML file example:
```yaml
task: detection  # Task to perform (e.g., segmentation, detection)
input_type: image  # Input type (e.g., image or lidar)
id: batch_id  # Batch identifier

# All models and datasets defined will be evaluated all-vs-all

model:
  - id: "model_id"  # Model identifier, if path is a pattern, model basename will be added as suffix
    path: "/path/to/model.pth"  # Path to the trained model file. It can be a pattern to match multiple model files (which will be evaluated independently)
    path_is_pattern: false  # Whether the path is a pattern or not
    format: torch  # Model format (e.g., torch, tensorflow)
    ontology: "/path/to/model_ontology.json"  # Path to the model output ontology JSON
    cfg: "/path/to/model_config.json"  # Path to the model configuration JSON
  - id: "another_model_id"
    # ...
  - id: "yet_another_model_id"
    # ...

dataset:
  - id: "dataset_id"  # Dataset identifier
    format: coco  # Dataset format (e.g., gaia, rellis3d, goose, generic, coco)
    dir: "/path/to/coco/dataset"  # (For COCO) Path to the COCO dataset directory
    fname: "/path/to/dataset.parquet"  # (For Gaia) Path to the dataset Parquet file
    split_dir: "/path/to/split_directory"  # (For Rellis3D/Wildscenes) Path to the directory containing split files
    train_dir: "/path/to/train_dataset_directory"  # (For Goose/Generic) Train directory
    val_dir: "/path/to/val_dataset_directory"  # (For Goose/Generic) Validation directory
    test_dir: "/path/to/test_dataset_directory"  # (For Goose/Generic) Test directory
    data_suffix: "_image.jpg"  # (For Generic) Data suffix
    label_suffix: "_label.png"  # (For Generic) Label suffix
    ontology: "/path/to/dataset_ontology.json"  # (For Rellis3D/Generic) Path to dataset ontology
    split: test  # Dataset split to evaluate (e.g., train, val, test)
  - id: "another_dataset_id"
    # ...

outdir: "/path/to/output_directory"  # Path to output directory
overwrite: false  # Whether to overwrite existing output files or not
ontology_translation: "/path/to/ontology_translation.json"  # (Optional)
store_results_per_sample: false  # Whether to store the predictions for each sample
```