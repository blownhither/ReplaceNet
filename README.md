# ReplaceNet

Salient Object Replacement in 2D Images   
with direct supervision from image synthesis


### Datasets
- Super small dataset of SOD (Salient Objects Dataset).   
    Descriptions: http://elderlab.yorku.ca/SOD/#download      
    - http://elderlab.yorku.ca/SOD/SOD.zip
    - https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300-images.tgz
    - decompress into SOD/ and BSDS300/ 
    ![](https://i.ibb.co/py0tSKk/masks.png)  
    ```python
  import numpy as np
  from load_data import load_parsed_sod
  images, masks = load_parsed_sod()
  assert masks[0].dtype == np.bool
    ```
    
    
### Training
- harmonize.py
    - [Deep Image Harmonization](https://arxiv.org/pdf/1703.00069.pdf)


### Using Synthesizer

```python
from synthesize import Synthesizer
from load_data import load_parsed_sod
from matplotlib import pyplot as plt
s = Synthesizer()
images, mask = load_parsed_sod()
si = s.synthesize(image=images[223], mask=mask[223], reference_mask=mask[53])
plt.imshow(si)
plt.show()
```