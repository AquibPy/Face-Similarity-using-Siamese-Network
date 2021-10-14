# Face Similarity using Siamese Network

## Standard Classification vs. One Shot Classification

**Standard classification** is what nearly all classification models use. The input is fed into a series of layers, and in the end , the class probabilities are output. If you want to predict dogs from cats, you train the model on similar(but not same) dogs/cats pictures that you would expect during prediction time. Naturally, this requires that you have a dataset that is similar to what you would expect once you use the model for prediction.

**One Shot Classification** models, on the other hand, requires that you have just one training example of each class you want to predict on. The model is still trained on several instances, but they only have to be in the similar domain as your training example.

A nice example would be facial recognition. You would train a One Shot classification model on a dataset that contains various angles , lighting , etc. of a few people. Then if you want to recognise if a person X is in an image, you take one single photo of that person, and then ask the model if that person is in the that image(note, the model was not trained using any pictures of person X).

## Siamese Networks

Siamese networks are a special type of neural network architecture. Instead of a model learning to classify its inputs, the neural networks learns to differentiate between two inputs. It learns the similarity between them.
A Siamese networks consists of two identical neural networks, each taking one of the two input images. The last layers of the two networks are then fed to a contrastive loss function , which calculates the similarity between the two images.

![](https://hackernoon.com/hn-images/1*XzVUiq-3lYFtZEW3XfmKqg.jpeg)

There are two sister networks, which are identical neural networks, with the exact same weights.

Each image in the image pair is fed to one of these networks.

## Contrastive Loss function

The objective of the siamese architecture is not to classify input images, but to differentiate between them. So, a classification loss function (such as cross entropy) would not be the best fit. Instead, this architecture is better suited to use a contrastive function. Intuitively, this function just evaluates how well the network is distinguishing a given pair of images.

The contrastive loss function is given as follows:
![](https://hackernoon.com/hn-images/1*tzGB6D97tHWR_-NJ8FKknw.jpeg)

where **Dw** is defined as the **euclidean distance** between the outputs of the sister siamese networks. Mathematically the **euclidean distance** is :

![](https://hackernoon.com/hn-images/1*6JCpYpYVJnpgYwupVIHSpg.jpeg)

where **Gw** is the output of one of the sister networks. **X1** and ****X2** is the input data pair.

### Equation Explanation

**Y** is either 1 or 0. If the inputs are from the same class , then the value of Y is 0 , otherwise Y is 1

**max()** is a function denoting the bigger value between 0 and **m-D**w.

**m** is a **margin value** which is greater than 0. Having a margin indicates that dissimilar pairs that are beyond this margin will not contribute to the loss. This makes sense, because you would only want to optimise the network based on pairs that are actually dissimilar , but the network thinks are fairly similar.

## Training the Siamese Network

The training process of a siamese network is as follows:

1) Pass the first image of the image pair through the network.
2) Pass the 2nd image of the image pair through the network.
3) Calculate the loss using the ouputs from 1 and 2.
4) Back propagate the loss to calculate the gradients.
5) Update the weights using an optimiser.

# Results

![](https://hackernoon.com/hn-images/1*N1p1_p5AU_7Ki1jPrsFhsw.png)