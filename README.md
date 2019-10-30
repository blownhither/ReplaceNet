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
  from load_data import load_parsed_data
  images, masks = load_parsed_data()
  assert masks[0].dtype == np.bool

    ```