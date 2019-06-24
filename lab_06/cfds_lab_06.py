
# coding: utf-8

# In[1]:


#-*- coding: utf-8 -*-


# <img align="right" style="max-width: 200px; height: auto" src="images/cfds_logo.png">
# 
# ###  Lab 06 - "Supervised Deep Learning - CNNs"
# 
# Chartered Financial Data Scientist (CFDS), Spring Term 2019

# In the fifth lab you learned about how to utilize an **supervised** (deep) machine learning technique namely the **Artificial Neural Network (ANN)** algorithm. 
# 
# In this sixth lab we will learn how to enhance ANNs using PyTorch to classify even more complex images. Therefore, we use a special type of deep neural networks referred to **Convolutional Neural Networks (CNNs)**. CNNs encompass the ability to take advantage of the hierarchical pattern in data and assemble more complex patterns using smaller and simpler patterns. Therefore, CNNs are able to learn a set of discriminative features 'pattern' and subsequently utilize the learned pattern to classify the content of an image.  
# 
# In this lab, we will implement and use a CNN to classify tiny images into categories such as airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks.

# <img align="center" style="max-width: 700px" src="images/classification.png">

# In case you experiencing any difficulties with the lab content or have any questions pls. don't hesitate to Marco Schreyer (marco.schreyer@unisg.ch).

# ### Lab Objectives:

# After today's lab you should be able to:
# 
# > 1. Understand the **basic concepts, intuitions and major building blocks** of convolutional neural networks.
# > 2. Know how to use Python's **PyTorch library** to train and evaluate neural network based models.
# > 3. Understand how to apply neural networks to **classify images** images based on their content into distinct categories.
# > 4. Know how to **interpret the detection results** of the network as well as its **reconstruction loss**.

# ### Step 0: Setup of the Jupyter Notebook Environment

# Similar to the previous labs, we need to import a couple of Python libraries that allow for data analysis and data visualization. We will mostly use the PyTorch, Numpy, Sklearn, Matplotlib, Seaborn and a few utility libraries throughout the course of this lab:

# In[2]:


# import standard python libaries
import os
from datetime import datetime
import numpy as np


# Import python machine / deep learning libraries:

# In[3]:


# import the PyTorch deep learning libary
import torch, torchvision
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable


# Import the sklearn classification metrics:

# In[4]:


# import sklearn classification evaluation library
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix


# Import python plotting libraries:

# In[5]:


# import matplotlib, seaborn, and PIL data visualization libary
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


# Enable notebook matplotlib inline plotting:

# In[6]:


#get_ipython().run_line_magic('matplotlib', 'inline')


# Create notebook folder structure to store the data as well as the trained neural network models:

# In[7]:


if not os.path.exists('./data'): os.makedirs('./data')  # create data directory
if not os.path.exists('./models'): os.makedirs('./models')  # create trained models directory


# ### Step 1.0: Dataset Download and Data Assessment

# The **CIFAR-10 database** (**C**anadian **I**nstitute **F**or **A**dvanced **R**esearch) is a collection of images that are commonly used to train machine learning and computer vision algorithms. The database is widely used for training and testing in the field of machine learning.

# <img align="center" style="max-width: 500px; height: 500px" src="images/cifar10.png">
# 
# (Source: https://www.kaggle.com/c/cifar-10)

# Further details on the dataset can be obtained via: *Krizhevsky, A., 2009. "Learning Multiple Layers of Features from Tiny Images",  
# ( https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf )."*

# In[67]:


cifar10_classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# The CIFAR-10 database contains 60,000 32x32 color images (50,000 training images and 10,000 validation images). The collection of images encompasses 10 different classes that represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. There are 6,000 images of each class. The CIFAR-10 is a good dataset that can be used to teach a computer how to recognize objects.
# 
# Let's download, transform and inspect the training images of the dataset.

# First we will define the directory we aim to store the training data:

# In[9]:


train_path = './data/train_cifar10'


# Now, let's download the training data accordingly:

# In[10]:


# define pytorch transformation into tensor format
transf = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# download and transform training images
cifar10_train_data = torchvision.datasets.CIFAR10(root=train_path, train=True, transform=transf, download=True)


# Verify the volume of training images downloaded:

# In[11]:


# get the length of the training data
len(cifar10_train_data)


# Furthermore, let's investigate a couple of the training images:

# In[88]:


# set (random) image id
image_id = 1111

# retrieve image exhibiting the image id
cifar10_train_data[image_id]


# Ok, that doesn't seem right :). Let's now seperate the image from its label information:

# In[89]:


cifar10_train_image, cifar10_train_label = cifar10_train_data[image_id]


# Great, let's now visually inspect our sample image: 

# In[90]:


# define tensor to image transformation
trans = torchvision.transforms.ToPILImage()

# set image plot title 
plt.title('Example: {}, Label: {}'.format(str(image_id), str(cifar10_classes[cifar10_train_label])))

# un-normalize cifar 10 image sample
cifar10_train_image_plot = cifar10_train_image / 2.0 + 0.5

# plot mnist cifar 10 image sample
plt.imshow(trans(cifar10_train_image_plot))


# Awsome, right? Let's now decide on were we want to store the evaluation data:

# In[15]:


eval_path = './data/eval_cifar10'


# And download the evaluation data accordingly:

# In[16]:


# define pytorch transformation into tensor format
transf = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# download and transform validation images
cifar10_eval_data = torchvision.datasets.CIFAR10(root=eval_path, train=False, transform=transf, download=True)


# Verify the volume of validation images downloaded:

# In[17]:


# get the length of the training data
len(cifar10_eval_data)


# ### Step 2.0 Neural Network Implementation

# In this section we will implement the architecture of the neural network we want to utilize in order to classify the MNIST images of handwritten digits.

# <img align="center" style="max-width: 900px" src="images/process.png">

# Our neural network, which we name 'MNISTNet' consists of three ** fully-connected layers** (including an “input layer” and two hidden layers). Furthermore, the MNISTNet should encompass the following number of neurons per layer: 100 (layer 1), 50 (layer 2) and 10 (layer 3). Meaning the first layer consists of 100 neurons, the second layer of 50 neurons and third layer of 10 neurons (the number of digit classes we aim to classify.

# In[18]:


# implement the MNISTNEt network architecture
class CIFAR10Net(nn.Module):
    
    # define the class constructor
    def __init__(self):
        
        # call super class constructor
        super(CIFAR10Net, self).__init__()
        
        # specify convolution layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0)
        
        # define max-pooling layer 1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # specify convolution layer 2
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        
        # define max-pooling layer 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # specify fc layer 1 - in 16 * 5 * 5, out 120
        self.linear1 = nn.Linear(16 * 5 * 5, 120, bias=True) # the linearity W*x+b
        self.relu1 = nn.ReLU(inplace=True) # the non-linearity
        
        # specify fc layer 2 - in 120, out 84
        self.linear2 = nn.Linear(120, 84, bias=True) # the linearity W*x+b
        self.relu2 = nn.ReLU(inplace=True) # the non-linarity
        
        # specify fc layer 3 - in 84, out 10
        self.linear3 = nn.Linear(84, 10) # the linearity W*x+b
        
        # add a softmax to the last layer
        self.logsoftmax = nn.LogSoftmax(dim=1) # the softmax
        
    # define network forward pass
    def forward(self, images):
        
        # define conv layer 1 forward pass
        x = self.pool1(self.relu1(self.conv1(images)))
        
        # define conv layer 2 forward pass
        x = self.pool2(self.relu2(self.conv2(x)))
        
        # reshape image pixels
        x = x.view(-1, 16 * 5 * 5)
        
        # define fc layer 1 forward pass
        x = self.relu1(self.linear1(x))
        
        # define fc layer 2 forward pass
        x = self.relu2(self.linear2(x))
        
        # define layer 3 forward pass
        x = self.logsoftmax(self.linear3(x))
        
        # return forward pass result
        return x


# You may have noticed that we applied a few more layers (compared to the MNIST example outlined in lab 05). These layers are referred to as **convolutional** and **max-pooling** layers are specifically designed to learn a set of features "patterns" from the processed images. The learned features are then be used by the subsequent non-linear layers to classify the images according to the then different classes contained in the CIFAR-10 dataset. 
# 
# In the following we will have a detailed look into the functionality and dimension of each layer. We will start with the input that we will provide to the network consisting of images that are 3x32x32, i.e., 3 channels (red, green, blue) each of size 32x32 pixels.
# 
# ** First Convolutional Layer: ** The first convolutional layer expects 3 input channels and will convolve 6 filters each of size 3x5x5. Since padding is set to 0 and stride is set to 1, the output size becomes 6x28x28, because (32 - 5) + 1 = 28. This layer exhibits ((5 x 5 x 3) + 1) x 6 = 456 parameter. 
# 
# ** First Max-Pooling Layer: ** The first down-sampling layer uses max-pooling with a 2x2 kernel and stride 2. This effectively drops the size from 6x28x28 to 6x14x14.
# 
# ** Second Convolutional Layer: ** The second convolutional layer expects 6 input channels and will convolve 16 filters each of size 6x5x5x. Since padding is set to 0 and stride is set 1, the output size is 16x10x10, because (14  - 5) + 1 = 10. This layer therefore has ((5 x 5 x 6) + 1 x 16) = 24,16 parameter.
# 
# ** Second Max-Pooling Layer: ** The second down-sampling layer uses max-pooling with 2x2 kernel and stride set to 2. This effectively drops the size from 16x10x10 to 16x5x5. 
# 
# ** Feature Flattening: ** The output of the final-max pooling layer needs to be flattened so that we can connect it to a fully connected layer. This is achieved using the `torch.Tensor.view` method. Setting the parameter of the method to `-1` will automatically infer the number of rows required to handle the mini-batch size of the data. 
# 
# ** First Fully-Connected Layer: ** The first uses "Rectified Linear Units" (ReLU) activation functions to learn potential nonlinear relationships evident in the data. The layer consists of 120 neurons, thus in total exhibits ((16 x 5 x 5) + 1) x 120 = 48,120 parameter. 
# 
# ** Second Fully-Connected Layer: ** The output of the first fully-connected layer is then transferred to second fully-connected layer. The layer consists of 84 neurons equipped with ReLu activation functions, this in total exhibits (120 + 1) x 84 = 10,164 parameter. 
# 
# 
# ** Output Layer: ** The output of the second fully-connected layer is then transferred to the output-layer (third fully-connected layer). The output layer is equipped with a softmax (that you learned about in the previous lab 05) and is made up of ten neurons, one for each object class contained in the CIFAR-10 dataset. This layer exhibits (84 + 1) x 10 = 850 parameter.
# 
# 
# As a result our CIFAR-10 convolutional neural exhibits a total of 456 + 2,416 + 48,120 + 10,164 + 850 = 62,006 parameter.
# 
# (Source: https://www.stefanfiott.com/machine-learning/cifar-10-classifier-using-cnn-in-pytorch/)

# Now, that we have implemented our first neural network we are ready to instantiate a network model to be trained:

# In[19]:


model = CIFAR10Net()


# Once the model is initialized we can visualize the model structure and review the implemented network architecture by execution of the following cell:

# In[20]:


# print the initialized architectures
print('[LOG] CIFAR10Net architecture:\n\n{}\n'.format(model))


# Looks like intended? Brilliant!

# Now that we have implemented the CIFAR10Net we are ready to train the network. However, prior to starting the training, we need to define an apropriate loss function. Remember, we aim to train our model to learn a set of model parameters $\theta$ that minimize the classification error of the true class $c^{i}$ of a given CIFAR-10 image $x^{i}$ and its predicted class $\hat{c}^{i} = f_\theta(x^{i})$ as faithfully as possible. 
# 
# Thereby, the training objective is to learn a set of optimal model parameters $\theta^*$ that optimize $\arg\min_{\theta} \|C - f_\theta(X)\|$ over all training images in the MNIST dataset. To achieve this optimization objective, one typically minimizes a loss function $\mathcal{L_{\theta}}$ as part of the network training. In this lab we use the **'Negative Log Likelihood (NLL)'** loss, defined by:
# 
# <center> $\mathcal{L}^{NLL}_{\theta} (c_i, \hat c_i) = - \frac{1}{N} \sum_{i=1}^N \log (\hat{c}_i) $, </center>

# for a set of $n$-CIFAR-10 images $x^{i}$, $i=1,...,n$ and their respective predicted class labels $\hat{c}^{i}$. This is summed for all the correct classes. 
# 
# Let's have a look at a brief example:

# <img align="center" style="max-width: 600px" src="images/loss.png">

# During trainig the NLL loss will penalize models that result in a high classification error between the predicted class labels $\hat{c}^{i}$ and their respective true class label $c^{i}$. Luckily, an implementation of the NLL loss is already available in PyTorch! It can be instantiated "off-the-shelf" via the execution of the following PyTorch command:

# In[21]:


# define the optimization criterion / loss function
nll_loss = nn.NLLLoss()


# Based on the loss magnitude of a certain mini-batch PyTorch automatically computes the gradients. But even better, based on the gradient, the library also helps us in the optimization and update of the network parameters $\theta$.
# 
# We will use the **Stochastic Gradient Descent (SGD) optimization** and set the `learning-rate to 0.001`. Each mini-batch step the optimizer will update the model parameters $\theta$ values according to degree of classification error (the NLL loss).

# In[22]:


# define learning rate and optimization strategy
learning_rate = 0.001
optimizer = optim.SGD(params=model.parameters(), lr=learning_rate)


# Now that we have successfully implemented and defined the three CNN building blocks let's take some time to review the `CIFAR10Net` model definition as well as the `loss`. Please, read the above code and comments carefully and don't hesitate to let us know any questions you might have.

# ### Step 3.0. Training the Neural Network Model

# In this section, we will train our neural network model (as implemented in section above) using the transformed images of handwritten digits. More specifically, we will have a detailed look into the distinct training steps as well as how to monitor the training progress.

# #### Step 3.1. Preparing the Network Training

# So far we have pre-processed the dataset, implemented the CNN and defined the classification error. Let's now start to train a corresponding model for **20 epochs** and a **mini-batch size of 128** CIFAR-10 images per batch. This implies that the whole dataset will be fed to the CNN 20 times in chunks of 128 images yielding to **391 mini-batches** (50.000 training images / 128 images per mini-batch) per epoch. After the processing of each mini-batch the parameters of the network will be updated. 

# In[73]:


# specify the training parameters
num_epochs = 20 # number of training epochs
mini_batch_size = 4 # size of the mini-batches


# Furthermore, lets specifiy and instantiate a corresponding PyTorch data loader that feeds the image tensors to our neural network:

# In[24]:


cifar10_train_dataloader = torch.utils.data.DataLoader(cifar10_train_data, batch_size=mini_batch_size, shuffle=True)


# #### Step 3.2. Running the Network Training

# Finally, we start training the model. The training procedure of each mini-batch is performed as follows: 
# 
# >1. do a forward pass through the CIFAR10Net network, 
# >2. compute the negative log likelihood classification error $\mathcal{L}^{NLL}_{\theta}(c^{i};\hat{c}^{i})$, 
# >3. do a backward pass through the CIFAR10Net network, and 
# >4. update the parameters of the network $f_\theta(\cdot)$.
# 
# To ensure learning while training our CNN model we will monitor whether the loss decreases with progressing training. Therefore, we obtain and evaluate the classification performance of the entire training dataset after each training epoch. Based on this evaluation we can conclude on the training progress and whether the loss is converging (indicating that the model might not improve any further).
# 
# The following elements of the network training code below should be given particular attention:
#  
# >- `loss.backward()` computes the gradients based on the magnitude of the reconstruction loss,
# >- `optimizer.step()` updates the network parameters based on the gradient.

# In[25]:


# init collection of training epoch losses
train_epoch_losses = []

# set the model in training mode
model.train()

# train the MNISTNet model
for epoch in range(num_epochs):
    
    # init collection of mini-batch losses
    train_mini_batch_losses = []
    
    # iterate over all-mini batches
    for i, (images, labels) in enumerate(cifar10_train_dataloader):
        
        # convert images to torch tensor
        images = Variable(images)
        
        # convert labels to torch tensor
        labels = Variable(labels)
        
        # run forward pass through the network
        output = model(images)
        
        # reset graph gradients
        model.zero_grad()
        
        # determine classification loss
        loss = nll_loss(output, labels)
        
        # run backward pass
        loss.backward()
        
        # update network paramaters
        optimizer.step()
        
        # collect mini-batch reconstruction loss
        train_mini_batch_losses.append(loss.data.item())

    # determine mean min-batch loss of epoch
    train_epoch_loss = np.mean(train_mini_batch_losses)
    
    # print epoch loss
    now = datetime.utcnow().strftime("%Y%m%d-%H:%M:%S")
    print('[LOG {}] epoch: {} train-loss: {}'.format(str(now), str(epoch), str(train_epoch_loss)))
    
    # save model to local directory
    model_name = 'cifar10_model_epoch_{}.pth'.format(str(epoch))
    torch.save(model.state_dict(), os.path.join("./models", model_name))
    
    # determine mean min-batch loss of epoch
    train_epoch_losses.append(train_epoch_loss)


# Upon successfull training let's visualize and inspect the training loss per epoch:

# In[26]:


# prepare plot
fig = plt.figure()
ax = fig.add_subplot(111)

# add grid
ax.grid(linestyle='dotted')

# plot the training epochs vs. the epochs' classification error
ax.plot(np.array(range(1, len(train_epoch_losses)+1)), train_epoch_losses, label='epoch loss (blue)')

# add axis legends
ax.set_xlabel("[training epoch $e_i$]", fontsize=10)
ax.set_ylabel("[Classification Error $\mathcal{L}^{NLL}$]", fontsize=10)

# set plot legend
plt.legend(loc="upper right", numpoints=1, fancybox=True)

# add plot title
plt.title('Training Epochs $e_i$ vs. Classification Error $L^{NLL}$', fontsize=10);


# Ok, fantastic. The training error converges nicely. We could definitly train the network a couple more epochs until the error converges. But let's stay with the 20 training epochs for now and continue with evaluating our trained model.

# ### Step 4.0. Evaluation of the Trained Neural Network Model

# Prior to evaluating our model let's load the best performing model. Remember, that we stored a snapshot of the model after each training epoch to our local model directory. We will now load the last snapshot saved.

# In[27]:


# restore pre-trained model snapshot
best_model_name = "cifar10_model_epoch_19.pth"

# init pre-trained model class
best_model = CIFAR10Net()

# load pre-trained models
best_model.load_state_dict(torch.load(os.path.join("models", best_model_name)))


# Let's inspect if the model was loaded successfully: 

# In[28]:


# set model in evaluation mode
best_model.eval()


# In order to evaluate our trained model we need to feed the CIFAR10 images reserved for evaluation (the images that we didn't use as part of the training process) through the model. Therefore, let's again define a corresponding PyTorch data loader that feeds the image tensors to our neural network: 

# In[80]:


cifar10_eval_dataloader = torch.utils.data.DataLoader(cifar10_eval_data, batch_size=10000, shuffle=False)


# We will now evaluate the trained model using the same mini batch approach as we did throughout the network training and derive the mean negative log likelihood loss of the mini-batches:

# In[81]:


# init collection of mini-batch losses
eval_mini_batch_losses = []

# iterate over all-mini batches
for i, (images, labels) in enumerate(cifar10_eval_dataloader):

    # convert images to torch tensor
    images = Variable(images)

    # convert labels to torch tensor
    labels = Variable(labels)

    # run forward pass through the network
    output = model(images)

    # determine classification loss
    loss = nll_loss(output, labels)

    # collect mini-batch reconstruction loss
    eval_mini_batch_losses.append(loss.data.item())

# determine mean min-batch loss of epoch
eval_loss = np.mean(eval_mini_batch_losses)

# print epoch loss
now = datetime.utcnow().strftime("%Y%m%d-%H:%M:%S")
print('[LOG {}] eval-loss: {}'.format(str(now), str(eval_loss)))


# Ok, great. The evaluation loss looks in-line with our training loss. Let's now inspect the a few sample predictions in order to get an impression of the model quality. Therefore, we will again pick a random image of our evaluation dataset and retrieve its PyTorch tensor as well as the corresponding label:

# In[82]:


# set (random) image id
image_id = 777

# retrieve image exhibiting the image id
cifar10_eval_image, cifar10_eval_label = cifar10_eval_data[image_id]


# Let's now inspect the true class of the image we selected:

# In[83]:


cifar10_classes[cifar10_eval_label]


# Ok, the randomly selected image should contain a two (2). Let's inspect the image accordingly:

# In[84]:


# define tensor to image transformation
trans = torchvision.transforms.ToPILImage()

# set image plot title 
plt.title('Example: {}, Label: {}'.format(str(image_id), str(cifar10_classes[cifar10_eval_label])))

# un-normalize cifar 10 image sample
cifar10_eval_image_plot = cifar10_eval_image / 2.0 + 0.5

# plot cifar 10 image sample
plt.imshow(trans(cifar10_eval_image_plot))


# Ok, let's compare the true label with the prediction of our model:

# In[85]:


cifar10_eval_image.unsqueeze(0).shape
best_model(cifar10_eval_image.unsqueeze(0))


# We can even determine the likelihood of the most probable class:

# In[79]:


cifar10_classes[torch.argmax(model(Variable(cifar10_eval_image.unsqueeze(0))), dim=1).item()]


# Let's now obtain the predictions for all the CIFAR-10 images of the evaluation data:

# In[63]:


predictions = torch.argmax(model(iter(cifar10_eval_dataloader).next()[0]), dim=1)


# Furthermore, let's obtain the overall classifcation accuracy:

# In[64]:


metrics.accuracy_score(cifar10_eval_data.targets, predictions)


# Let's also inspect the confusion matrix to determine major sources of missclassification

# In[72]:


# determine classification matrix of the predicted and target classes
mat = confusion_matrix(cifar10_eval_data.targets, predictions)

# plot corresponding confusion matrix
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, cmap='YlOrRd_r', xticklabels=cifar10_classes, yticklabels=cifar10_classes)
plt.title('CIFAR-10 classification matrix')
plt.xlabel('[true label]')
plt.ylabel('[predicted label]');


# Ok, we can easily see that our current model is confusiong the digits 3 and 5 as well the digits 9 and 4 quite often. When looking at the corresponding images this makes sense right?   

# ### Exercises:

# We recommend you to try the following exercises as part of the lab:
# 
# **1. Train the network a couple more epochs and evaluate its prediction accuracy.**
# 
# > Increase the number of training epochs up to 50 epochs and re-run the network training. Load and evaluate the model exhibiting the lowest training loss. What kind of behavior in terms of prediction accuracy can be observed with increasing the training epochs?

# **2. Evaluaton of "shallow" vs. "deep" neural network architectures.**
# 
# > In addition to the architecture of the lab notebook, evaluate further (more shallow as well as more deep) neural network architectures by (1) either removing or adding layers to the network and/or (2) increasing/decreasing the number of neurons per layer. Train a model (using the architectures you selected) for at least 50 training epochs. Analyse the prediction performance of the trained models in terms of training time and prediction accuracy. 

# ### Lab Summary:

# In this fifth lab, a step by step introduction into **design, implementation, training and evaluation** of neural networks to classify images of handwritten digits is presented. The code and exercises presented in this lab may serves as a starting point for developing more complex, more deep and tailored **neural networks**.

# You may want to execute the content of your lab outside of the jupyter notebook environment e.g. on compute node or server. The cell below converts the lab notebook into a standalone and executable python script.

# In[ ]:


get_ipython().system('jupyter nbconvert --to script cfds_lab_06.ipynb')


# **Note:** In order to execute the statement above and convert your lab notebook to a regular Python script you first need to install the nbconvert Python package e.g. using the pip package installer. 
