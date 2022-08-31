# -*- coding: utf-8 -*-
"""Programming Assignment-04-CS4732-Troy Cope.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tRAoqX7DeaJs4Jr48qypdimstoQ_rGEO

# Using DCGANs to generate face images

 

First, let's understand the technique we will leverage to generate an image using a set of 100 random numbers. We will first convert noise into a shape of batch size x 100 x 1 x 1. The reason for appending additional channel information in DCGANs and not doing it in the GAN section is that we will leverage CNNs in this section, which requires inputs in the form of batch size x channels x height x width. Next, we convert the generated noise into an image by leveraging ConvTranspose2d.

 

As we learned previously that the Image Segmentation does the opposite of a convolution operation, which is to take input with a smaller feature map size (height x width) and upsample it to that of a larger size using a predefined kernel size, stride, and padding. This way, we would gradually convert a vector from a shape of batch size x 100 x 1 x 1 into a shape of batch size x 3 x 64 x 64. With this, we have taken a random noise vector of size 100 and converted it into an image of a face. 

 

With this understanding, let's now build a model to generate images of faces:
"""

!wget https://www.dropbox.com/s/rbajpdlh7efkdo1/male_female_face_images.zip

!unzip male_female_face_images.zip

"""Import the relevant packages:"""

!pip install -q --upgrade torch_snippets

from torch_snippets import *

import torchvision

from torchvision import transforms

import torchvision.utils as vutils

import cv2, numpy as np, pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"

"""Define the dataset and dataloader:
 

Ensure that we crop the images so that we retain only the faces and discard additional details in the image. First, we will download the cascade filter, which will help in identifying faces within an image:
"""

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

"""Create a new folder and dump all the cropped face images into the new folder:"""

!mkdir cropped_faces

images = Glob('/content/females/*.jpg') + Glob('/content/males/*.jpg')

for i in range(len(images)):
  img = read(images[i],1)

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  faces = face_cascade.detectMultiScale(gray, 1.3, 5)

  for (x,y,w,h) in faces:

    img2 = img[y:(y+h),x:(x+w),:]

    cv2.imwrite('cropped_faces/'+str(i)+'.jpg', cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))

"""Note that, by cropping and keeping faces only, we are retaining only the information that we want to generate.

 

Specify the transformation to perform on each image:

 
"""

transform=transforms.Compose([

transforms.Resize(64),

transforms.CenterCrop(64),

transforms.ToTensor(),

transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

"""Define the Faces dataset class:"""

class Faces(Dataset):

  def __init__(self, folder):

    super().__init__()

    self.folder = folder

    self.images = sorted(Glob(folder))

  def __len__(self):

    return len(self.images)

  def __getitem__(self, ix):

    image_path = self.images[ix]

    image = Image.open(image_path)

    image = transform(image)

    return image

"""Create the dataset object – ds:"""

ds = Faces(folder='cropped_faces/')

"""Define the dataloader class as follows:"""

dataloader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=8)

"""
Define weight initialization so that the weights have a smaller spread:"""

def weights_init(m):

  classname = m.__class__.__name__

  if classname.find('Conv') != -1:

    nn.init.normal_(m.weight.data, 0.0, 0.02)

  elif classname.find('BatchNorm') != -1:

    nn.init.normal_(m.weight.data, 1.0, 0.02)

    nn.init.constant_(m.bias.data, 0)

"""Define the Discriminator model class, which takes an image of a shape of batch size x 3 x 64 x 64 and predicts whether it is real or fake:"""

class Discriminator(nn.Module):

  def __init__(self):

    super(Discriminator, self).__init__()

    self.model = nn.Sequential(

    nn.Conv2d(3,64,4,2,1,bias=False),

    nn.LeakyReLU(0.2,inplace=True),

    nn.Conv2d(64,64*2,4,2,1,bias=False),

    nn.BatchNorm2d(64*2),

    nn.LeakyReLU(0.2,inplace=True),

    nn.Conv2d(64*2,64*4,4,2,1,bias=False),

    nn.BatchNorm2d(64*4),

    nn.LeakyReLU(0.2,inplace=True),

    nn.Conv2d(64*4,64*8,4,2,1,bias=False),

    nn.BatchNorm2d(64*8),

    nn.LeakyReLU(0.2,inplace=True),

    nn.Conv2d(64*8,1,4,1,0,bias=False),

    nn.Sigmoid()

    )

    self.apply(weights_init) 

  def forward(self, input): 

    return self.model(input)

"""Obtain a summary of the defined model:"""

!pip install torch_summary

from torchsummary import summary

discriminator = Discriminator().to(device)

summary(discriminator,torch.zeros(1,3,64,64));

"""Define the Generator model class that generates fake images from an input of shape batch size x 100 x 1 x 1:"""

class Generator(nn.Module):

  def __init__(self):

    super(Generator,self).__init__()

    self.model = nn.Sequential(

    nn.ConvTranspose2d(100,64*8,4,1,0,bias=False,),

    nn.BatchNorm2d(64*8),

    nn.ReLU(True),

    nn.ConvTranspose2d(64*8,64*4,4,2,1,bias=False),

    nn.BatchNorm2d(64*4),

    nn.ReLU(True),

    nn.ConvTranspose2d( 64*4,64*2,4,2,1,bias=False),

    nn.BatchNorm2d(64*2),

    nn.ReLU(True),

    nn.ConvTranspose2d( 64*2,64,4,2,1,bias=False),

    nn.BatchNorm2d(64),

    nn.ReLU(True),

    nn.ConvTranspose2d( 64,3,4,2,1,bias=False),

    nn.Tanh()

    )

    self.apply(weights_init)

  def forward(self,input): return self.model(input)

"""Obtain a summary of the defined model:"""

generator = Generator().to(device)

summary(generator,torch.zeros(1,100,1,1))

"""
Define the functions to train the generator (generator_train_step) and the discriminator (discriminator_train_step):

In the following code, we are performing a .squeeze operation on top of the prediction as the output of the model has a shape of batch size x 1 x 1 x 1 and it needs to be compared to a tensor that has a shape of batch size x 1."""

def discriminator_train_step(real_data, fake_data):

  d_optimizer.zero_grad()

  prediction_real = discriminator(real_data)

  error_real = loss(prediction_real.squeeze(), \

  torch.ones(len(real_data)).to(device))

  error_real.backward()

  prediction_fake = discriminator(fake_data)

  error_fake = loss(prediction_fake.squeeze(), \

  torch.zeros(len(fake_data)).to(device))

  error_fake.backward()

  d_optimizer.step()

  return error_real + error_fake

 

def generator_train_step(fake_data):

  g_optimizer.zero_grad()

  prediction = discriminator(fake_data)

  error = loss(prediction.squeeze(), \

  torch.ones(len(real_data)).to(device))

  error.backward()

  g_optimizer.step()

  return error

"""Create the generator and discriminator model objects, the optimizers, and the loss function of the discriminator to be optimized:"""

discriminator = Discriminator().to(device)

generator = Generator().to(device)

loss = nn.BCELoss()

d_optimizer = optim.Adam(discriminator.parameters(), \

lr=0.0002, betas=(0.5, 0.999))

g_optimizer = optim.Adam(generator.parameters(), \

lr=0.0002, betas=(0.5, 0.999))

"""Run the models over increasing epochs:
 

Loop through 25 epochs over the dataloader function defined in step 3:

Run the models over increasing epochs:
 

Loop through 25 epochs over the dataloader function defined in step 3:
 

Load real data (real_data) and generate fake data (fake_data) by passing through the generator network:


Note that the major difference between vanilla GANs and DCGANs when generating real_datais that we did not have to flatten real_data in the case of DCGANs as we are leveraging CNNs.

 

Train the discriminator using the discriminator_train_step function defined in step 7:
 

Generate a new set of images (fake_data) from the noisy data (torch.randn(len(real_data))) and train the generator using the generator_train_step function defined in step 7:

"""

log = Report(25)

for epoch in range(25):

  N = len(dataloader)

  for i, images in enumerate(dataloader):
    real_data = images.to(device)   

    fake_data = generator(torch.randn(len(real_data), 100, 1, 1).to(device)).to(device)

    fake_data = fake_data.detach()

    d_loss=discriminator_train_step(real_data, fake_data)

    fake_data = generator(torch.randn(len(real_data), 100, 1, 1).to(device)).to(device)

    g_loss = generator_train_step(fake_data)

    log.record(epoch+(1+i)/N, d_loss=d_loss.item(), g_loss=g_loss.item(), end='\r')

  log.report_avgs(epoch+1)

# fake_data = generator(torch.randn(len(real_data), 100, 1, 1).to(device)).to(device)

# fake_data = fake_data.detach()

# d_loss=discriminator_train_step(real_data, fake_data)

# fake_data = generator(torch.randn(len(real_data), 100, 1, 1).to(device)).to(device)

# g_loss = generator_train_step(fake_data)

"""Record the losses:"""

# log.record(epoch+(1+i)/N, d_loss=d_loss.item(), g_loss=g_loss.item(), end='\r')

# log.report_avgs(epoch+1)

log.plot_epochs(['d_loss','g_loss'])

"""Note that in this setting, the variation in generator and discriminator losses does not follow the pattern that we have seen in the case of handwritten digit generation on account of the following:

We are dealing with bigger images (images that are 64 x 64 x 3 in shape when compared to images of 28 x 28 x 1 shape, which we have seen in the previous section).
Digits have fewer variations when compared to the features that are present in the image of a face.
Information in handwritten digits is available in only a minority of pixels when compared to the information in images of a face.
Once the training process is complete, generate a sample of images using the following code:
"""

generator.eval()

noise = torch.randn(64, 100, 1, 1, device=device)

sample_images = generator(noise).detach().cpu()

grid = vutils.make_grid(sample_images,nrow=8,normalize=True)

show(grid.cpu().detach().permute(1,2,0), sz=10, \

title='Generated images')