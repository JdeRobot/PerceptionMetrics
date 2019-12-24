---
layout: home
title: Converter
permalink: /functionality/converter/


sidebar:
  nav: "functionality"
---

This tool takes a dataset from a specific format and converts it to another format.
To complete this process, DetectionSuite needs the input dataset format (reader) and its class names and additionally 
the output dataset wanted. More options are available, like splitting the new dataset into 2 separated parts (train and test set).

### Command line use example

An example of config file would be:

```
    inputPath: /opt/datasets/weights/annotations/instances_val2017.json
    readerImplementation: COCO
    readerNames: /opt/datasets/names/coco.names
    writerImplementation: Pascal VOC
    inferencerNames: /opt/datasets/names/coco.names
    outputPath: /opt/datasets/output/new-dataset/
    writeImages: no
```

With the config file, change the directory to ``Tools/Converter`` inside build and run

```
    ./converter -c appConfig.yml
```

This will output the new converted dataset to the folder described in the configuration.

### GUI use video example

{% include video id="bOjt0v_h640" provider="youtube" %}

The above video demonstrates usage of **Converter** tool available in DatasetEvaluationApp, by converting *Pascal VOC dataset* to *COCO dataset* format. 
Conversion requires and input dataset with corresponding class names file, an output dataset, output folder and optionally a writer class names file.

`Map to writer classnames file` option is available. When this option is checked, the output dataset's class names file to which the input datasets class names will be mapped is necessary.
Mapping means matching synonyms in the detected objects class names. For instance, `motorbike` => `motorcycle`, `sofa` => `couch`, `airplane` => `aeroplane`, etc. 
After a successful conversion, all the mappings are printed along with discarded classes. A class may be discarded if the output dataset class names file doesn't contain of any such class.

This functionality can also be used for **filtering** classes out of a dataset in order to create a new dataset.

If `Map to writer classnames file` option is unchecked, then a new class names file will be generated containing all classes in the input dataset.
No classes are discarded in this case. 

Finally, after a successful conversion, [**Viewer**](/functionality/viewer/) tab can be used to see the converted dataset.