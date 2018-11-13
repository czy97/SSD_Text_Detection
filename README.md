# SSD_Text_Detection
This project is refered from the [SSD code](https://github.com/pengzhiliang/object-localization)

## Require：
- python 3.x  
- pytorch 0.4.1  
- opencv-python  
- Pillow

## Function:
- Define arbitrary hiddens layers as you want  
- Choose activation function between sigmoid and relu  
- BatchNorm  
- Dropout  
- Model storing(FullyConnectedNet.storeModel())  
- Model loading(FullyConnectedNet.loadModel())  
- Choose different update rules among sgd/sgd_momentum/rmsprop/adam  
- Seperated model definition module(codes.classifiers.fc_net.FullyConnectedNet) and updating module(codes.solver.Solver)

## Demo
mnist_classification.ipynb

## Postscript
The bestParams.pkl in the params directory I didn't upload here.  
If you need email me to chenzhengyang117@gmail.com
