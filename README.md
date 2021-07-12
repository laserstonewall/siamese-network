# siamese-network

`siamese-network` is a `PyTorch` implementation of Siamese neural networks for images, including proper image preparation.

In the typical application for a convolutional neural network (CNN), we are interested in classifying a number of different types of objects in images. ImageNet, for example, has 1000 image categories and > 1 million images of everyday objects such as photocopier, hot dog, brocolli, etc. Often we use transfer learning to retrain these networks to new object classes for new applications.

Siamese networks have a different objective. For a pair of images, the objective of a Siamese net is to determine the level of similarity of the two images. Each image enters an identical copy of the CNN that has had the fully connected final classification layers removed, so that the network output is the high dimensional feature extraction layer that is usually fed into the FC layers. The high dimensional features from image pairs are compared with a new set of fully connected layers. The final layer can output a binary classification (images contain same/different objects) or a similarity distance, depending on the choice of setup.

One of the major applications of Siamese neural networks is for one-shot learning. Here, a network is trained to learn when images are similar/dissimilar. For example, we train the network with a dataset consisting of many types of balls (basketball, baseball, raquetball, etc.). After training, we want to take a single image of a new type of ball (say it's never seen a tennis ball before), and use that to identify other tennis balls in new image data. 

This type of one-shot learning is very interesting for applications like facial recognition. Think about Apple's FaceID. It takes a few shots of your face from various angles, and from then on it can determine if new face images taken during the authentication process are you or someone else attempting to access your phone. Apple likely uses some sort of Siamese network underneath the hood to accomplish this, using a trained network to infer whether or not the new image is the phone's owner.

A sample workflow is contained in `nbs/Oxford-IIIT_Pet.ipynb`

### Installation

#### Create a new environment with required packges

To create an environment with all required packages capable of running the example notebook, use the provided `environment.yml` file to install a `conda` environment with the needed packages, including the `siamese` package contained in this repo:

```bash
conda env create -f environment.yml
```

Then run:

```bash
conda activate siamese
```

#### Install into existing `conda` environment

The only dependency for the core functionality of `siamese` are the `SiamesePairedDataset` and `SiameseNetwork` classes. These work directly in `PyTorch`, and can be imported and used as any normal `Dataset` and network would be. To install these directly to your existing environment, from the main project directory run:

```bash
conda activate <your environment here>
pip install .
```

### To Do

- Add user specified option for feature extraction layer comparison function
- Add additional example using only PyTorch