---
layout: home
title: Converter
permalink: /functionality/converter/


sidebar:
  nav: "docs"
---

<p align="center">
<kbd><a href="http://www.youtube.com/watch?feature=player_embedded&v=bOjt0v_h640" target="_blank"><img src="http://img.youtube.com/vi/bOjt0v_h640/0.jpg"
alt="IMAGE ALT TEXT HERE" width="480" height="360" border="8"/></a>
</kbd>
</p>

The above video demonstrates usage of converter tool available in DatasetEvaluationApp, by converting Pascal VOC dataset to COCO Dataset Format. 
Conversion requires and Input Dataset with corresponding classnames file, an output dataset, output folder and optionally a writer class names file.

If `Map to writer classnames file` option is checked, then the output dataset's classnames file is necessary to which the input datasets class names will be mapped.
Here mapping means matching synonyms. For Instance, motorbike => motorcycle, sofa => couch, airplane => aeroplane, etc. After successful conversion all the mappings are printed along with discarded classes. A class may be discarded if the output dataset classnames file doesn't consist of any such class.

This functionality can also be used for **filtering** classes out of a dataset in order to create a new dataset.

If `Map to writer classnames file` option is unchecked, then a new classnames file will be generated containing all classes in the input dataset.
No classes are discarded in this case. 

Finally, after successful conversion, Viewer tab is used to see the converted dataset.