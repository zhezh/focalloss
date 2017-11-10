## Focal Loss

```
Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal Loss for Dense Object Detection, 130(4), 485–491. https://doi.org/10.1016/j.ajodo.2005.02.022
```

Implementation for focal loss in tensorflow.

This focal loss is a little different from the original one described in paper. This one is for **multi-class** classification tasks other than binary classifications. 

The input are softmax-ed probabilities.