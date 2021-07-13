# siamese-network

`siamese-network` is a `PyTorch` implementation of Siamese neural networks for images, including proper image preparation.

In the typical application for a convolutional neural network (CNN), we are interested in classifying a number of different types of objects in images. ImageNet, for example, has 1000 image categories and > 1 million images of everyday objects such as photocopier, hot dog, brocolli, etc. Often we use transfer learning to retrain these networks to new object classes for new applications.

Siamese networks have a different objective. For a pair of images, the objective of a Siamese net is to determine the level of similarity of the two images. Each image enters an identical copy of the CNN that has had the fully connected final classification layers removed, so that the network output is the high dimensional feature extraction layer that is usually fed into the FC layers. The high dimensional features from image pairs are compared with a new set of fully connected layers. The final layer can output a binary classification (images contain same/different objects) or a similarity distance, depending on the choice of setup.

One of the major applications of Siamese neural networks is for one-shot learning. Here, a network is trained to learn when images are similar/dissimilar. For example, we train the network with a dataset consisting of many types of balls (basketball, baseball, raquetball, etc.). After training, we want to take a single image of a new type of ball (say it's never seen a tennis ball before), and use that to identify other tennis balls in new image data. 

This type of one-shot learning is very interesting for applications like facial recognition. Think about Apple's FaceID. It takes a few shots of your face from various angles, and from then on it can determine if new face images taken during the authentication process are you or someone else attempting to access your phone. Apple likely uses some sort of Siamese network underneath the hood to accomplish this, using a trained network to infer whether or not the new image is the phone's owner.

A sample workflow is contained in `nbs/Oxford-IIIT_Pet.ipynb`

### Installation

First run:

```bash
git clone https://github.com/laserstonewall/siamese-network
```

to pull the repository locally. Then either build and run the provided Docker container, or install directly on your development machine.

#### Build and run the provided Docker container

If you haven't already, [install Docker](https://docs.docker.com/get-docker/) on your system, and then also install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) following the instructions for Docker.

Once Docker and the NVIDIA Container Toolkit are installed, from the `siamese-network` main directory, run:

```bash
docker build --build-arg CUDA_VERSION=XXX -t nvidia_siamese:latest .
```
substituting in your version of CUDA for XXX. If you are unsure of your CUDA version, run `nvidia-smi` and it will tell you your CUDA version in the upper right. For example, if you have `CUDA Version: 10.2` the command is:

```bash
docker build --build-arg CUDA_VERSION=10.2 -t nvidia_siamese:latest .
```

The container will take several minutes to build. Once complete, the container can be started, with full GPU support and a running Jupyter notebook with the Oxford Pets example notebook with:

```bash
docker run -it --shm-size=2g -m=4g -p 9183:9183 --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all nvidia_siamese /bin/bash -c "source activate siamese && jupyter notebook --no-browser --port=9183 --ip=0.0.0.0 --allow-root"
```

Once started, the Jupyter notebook will display a URL + token, copy this and paste into the browser on your local development machine, and you should be 

#### Local installation: Create a new environment with required packages

To create an environment with all required packages capable of running the example notebook, use the provided `environment.yml` file to install a `conda` environment with the needed packages, including the `siamese` package contained in this repo:

```bash
conda env create -f environment.yml
```

Then run:

```bash
conda activate siamese
```

#### Local Installation: Install into existing `conda` environment

The only dependency for the core functionality of `siamese` are the `SiamesePairedDataset` and `SiameseNetwork` classes. These work directly in `PyTorch`, and can be imported and used as any normal `Dataset` and network would be. To install these directly to your existing environment, from the main project directory run:

```bash
conda activate <your environment here>
pip install .
```

### Using `siamese`

The core functionality of `siamese` is contained in `SiamesePairedDataset` and `SiameseNetwork`.

#### `SiamesePairedDataset`

As implemented here, Siamese networks take in pairs of images, and output a label predicting whether the image pair is from the same class or different classes. `SiamesePairedDataset` inherits from the base PyTorch `Dataset` class, adding specific functionality to generate image pairs and the sameness/difference label.

In order to use it, you'll need a `pandas` DataFrame with two columns to input to `SiamesePairedDataset`:

1. A column containing the full path to the individual images.
2. A column indicating the class the image belongs to.

For example, the `data` folder is two levels up from your current directory, you might end up with:

| Image                                      | Label |
| ------------------------------------------ | :---: |
| ../../data/images/Abyssinian_100.jpg       |   1   |
| ../../data/images/Abyssinian_100.jpg       |   1   |
| ...                                        |       |
| ../../data/images/yorkshire_terrier_99.jpg |   37  |
| ../../data/images/yorkshire_terrier_9.jpg  |   37  |

**Inputs**

- **data**: The `pandas` DataFrame with columns for image path and class described above.
- **path_col**: The column name in `data` DataFrame containing the image paths.
- **label_col**: The column name in `data` DataFrame containing the image class labels.
- **sampling_strategy**: How classes will be sampled to generate image pairs. If `'uniform'` all classes will be sampled with equal probability, regardless of the relative number of examples of each class. If `'proportional'` all classes will sampled according to their prevalence in the data. If `'custom'`, the probability of sampling each class must be specified according to the documentation below for the `class_prob` kwarg. 
- **class_prob**: Unless `sampling_strategy` is `'custom'` this should be `None`. If `'custom'` is used, this should be an array where each value corresponds to the class sampling probability, where the order of the classes is determined by the order of `data[label_col].unique()`.
- **transform**: The set of PyTorch transforms to use on the images for image augmentation. Should be assembled using PyTorch's `transforms.Compose` function. The pre-trained ResNets from PyTorch used here function best when their inputs are in the same range they were trained on, `[-1, 1]`. To ensure this is the case, it's often a good idea to have `transforms.Lambda(lambda x: (x - 0.5) / 0.5)` as the final transform in your `Compose` statement.

#### SiameseNetwork

The actual PyTorch neural network used for the Siamese network. The network uses an existing convolutional neural network, usually one of PyTorch's pretrained CNNs. Since the CNN is passed as an argument, a network with randomly initialized or pre-trained weights can be used. The network currently expects images with 3-channels, so 1-channel grayscale images need to be converted to 3-channel using the PyTorch transform `transforms.Grayscale(num_output_channels=3)` in the transform passed to `SiamesePairedDataset`.

**Inputs**

- **transfer_network**: A PyTorch CNN, with the fully connected layers intact, and usually with pre-trained weights initialized. For example, the Siamese network could be initialized with a pre-trained ResNet34 with: `model_ft = SiameseNetwork(models.resnet34(pretrained=True))`. 

### To Do

- Add user specified option for feature extraction layer comparison function
- Add additional example using only PyTorch