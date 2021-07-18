This is a project intended to create a new CNN Network for Image Classification. Fas14MNet can also be used for Multi-Label Image Classification and in this project we try to create a CNN that will work equally well with for Traditional AI architectures as well as a Federated Architecture and Fas14MNet helps us achieve that. Fas14MNet is inspired from the basic architecture of a ResNet50 model but is much lighter while having ~15M parameters compared to 23M+ parameters in a ResNet50 model.


Steps to run:

1. from fas14mnet.net import Fas14MNet

2. model = Fas14MNet(num_classes)