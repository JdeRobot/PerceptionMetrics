---
layout: home
title: Converter
permalink: /functionality/converter/


sidebar:
  nav: "docs"
---
**Converter** takes a dataset from a specific format and converts it to another format, given some parameter that specify 
how this process should be addressed.

{% include video id="bOjt0v_h640" provider="youtube" %}

The above video demonstrates usage of **Converter** tool available in DatasetEvaluationApp, by converting *Pascal VOC dataset* to *COCO dataset* format. 
Conversion requires and input dataset with corresponding class names file, an output dataset, output folder and optionally a writer class names file.

`Map to writer classnames file` option is available. When this option is checked, the output dataset's class names file to which the input datasets class names will be mapped is necessary.
Mapping means matching synonyms in the detected objects class names. For instance, `motorbike` => `motorcycle`, `sofa` => `couch`, `airplane` => `aeroplane`, etc. 
After a successful conversion, all the mappings are printed along with discarded classes. A class may be discarded if the output dataset class names file doesn't contain of any such class.

This functionality can also be used for **filtering** classes out of a dataset in order to create a new dataset.

If `Map to writer classnames file` option is unchecked, then a new class names file will be generated containing all classes in the input dataset.
No classes are discarded in this case. 

Finally, after a successful conversion, **Viewer** tab can be used to see the converted dataset.