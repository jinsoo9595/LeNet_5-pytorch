# Pytorch implementation of LeNet-5
This is an implementaiton of LeNet-5 in Pytorch. This code visualizes modules Loss, Feature Map, and Classification result(t-SNE) for the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).

> Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to document recognition." Proceedings of the IEEE, 86(11):2278-2324, November 1998. [LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)

   
   
   

## Characteristic
- Monitoring Loss (Visdom)
- Trained model export (ONNX)
- Feature map visualization
- Plot FC layer classification feature

   
   
   

## Performance
The table below shows models, dataset and performances

Model | Dataset | Top-1 | Top-5 
:----:| :------:| :----:|:-----:
LeNet| MNIST | 98.74% | - 


