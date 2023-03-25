---
title: Pneumonia Classification with PyTorch
date: 2023-03-03 14:10:00 +0800
categories: [Engineering, Deep Learning]
tags: [python, pytorch, convolutional neural networks]
render_with_liquid: false
---

Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus (purulent material), causing cough with phlegm or pus, fever, chills, and difficulty breathing. A variety of organisms, including bacteria, viruses and fungi, can cause pneumonia. An early detection and treatment to prevent progression might be crucial due to its fatality rate, which is 5 to 10 percent for hospitalized patients.

This study aims to provide a classification model trained with the X-Ray images provided in the (pneumonia detection challenge)[https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/] in Kaggle. Further information regarding the dataset can be found in the (*ChestX-ray8*)[https://arxiv.org/abs/1705.02315] paper.

## Importing Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import pydicom  # To read dicom files
# Modeling --------------------------
import torch
import torchvision
from torchvision import transforms  # For data augmentation & normalization
import torchmetrics # Easy metric computation
import pytorch_lightning as pl  
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
# -----------------------------------
from pathlib import Path    # For file path handling
import cv2  # For image resizing
from tqdm.notebook import tqdm  # For progress bar
import os
```

## Preprocessing and Loading the Data

```python
# Reading the data labels

labels = pd.read_csv("stage_2_train_labels.csv")
labels.head(10)
```





  <div id="df-7ff43fd1-70c0-4459-9e1d-3bd89ba36b84">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>patientId</th>
      <th>x</th>
      <th>y</th>
      <th>width</th>
      <th>height</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0004cfab-14fd-4e49-80ba-63a80b6bddd6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>00313ee0-9eaa-42f4-b0ab-c148ed3241cd</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00322d4d-1c29-4943-afc9-b6754be640eb</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>003d8fa0-6bf1-40ed-b54c-ac657f8495c5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00436515-870c-4b36-a041-de91049b9ab4</td>
      <td>264.0</td>
      <td>152.0</td>
      <td>213.0</td>
      <td>379.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>00436515-870c-4b36-a041-de91049b9ab4</td>
      <td>562.0</td>
      <td>152.0</td>
      <td>256.0</td>
      <td>453.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>00569f44-917d-4c86-a842-81832af98c30</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>006cec2e-6ce2-4549-bffa-eadfcd1e9970</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>00704310-78a8-4b38-8475-49f4573b2dbb</td>
      <td>323.0</td>
      <td>577.0</td>
      <td>160.0</td>
      <td>104.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>00704310-78a8-4b38-8475-49f4573b2dbb</td>
      <td>695.0</td>
      <td>575.0</td>
      <td>162.0</td>
      <td>137.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-7ff43fd1-70c0-4459-9e1d-3bd89ba36b84')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-7ff43fd1-70c0-4459-9e1d-3bd89ba36b84 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-7ff43fd1-70c0-4459-9e1d-3bd89ba36b84');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




The data includes 6 columns including patient ID, target variable, and information about the location of the pneumonia if it exists. Notice that there are multiple entries for some patients since pneumonia can be located at more than one segment of an X-Ray image. Therefore, they will only be relevant in the second part of this excercise.


```python
# Dropping the duplicate rows based on patient ID

labels = labels.drop_duplicates("patientId")
```


```python
# Defining import and export paths

ROOT_PATH = Path("stage_2_train_images/")
SAVE_PATH = Path("Processed/")
```


```python
# Writing a nested loop to read and visualize a sample from the dataset

fig, axis = plt.subplots(3, 3, figsize = (9, 9))
c = 0
for i in range(3):
    for j in range(3):
        
        # Load image
        patient_id = labels.patientId.iloc[c]
        dcm_path = ROOT_PATH/patient_id
        dcm_path = dcm_path.with_suffix(".dcm")
        dcm = pydicom.read_file(dcm_path).pixel_array
        
        # Print image
        label = labels["Target"].iloc[c]
        
        axis[i][j].imshow(dcm, cmap="bone")
        axis[i][j].set_title(label)
        c += 1
```


    
![Figure 1](/assets/img/content/230303/output_17_0.png)  
<p style="text-align: center;"><em>Figure 1. Images Before Augmentation</em></p>
    



```python
len_train = len(os.listdir(ROOT_PATH))
len_valid = len(os.listdir("stage_2_test_images/"))
print(len_train)
print(len_valid)
```

    26684
    3000
    


```python
# Resizing the images for an easy training
# Standardazing the pixel values by dividing by 255 
# Using tqdm to track the loading progress

sums, sums_squared = 0, 0

for c, patient_id in enumerate(tqdm(labels.patientId)):
    
    # Load image
    patient_id = labels.patientId.iloc[c]
    dcm_path = ROOT_PATH/patient_id
    dcm_path = dcm_path.with_suffix(".dcm")
    dcm = pydicom.read_file(dcm_path).pixel_array / 255  
    
    dcm_array = cv2.resize(dcm, (224, 224)).astype(np.float16)
    
    label = labels.Target.iloc[c]

    # Splitting the data into training and test

    train_or_test = "train" if c < 24000 else "test" 
        
    current_save_path = SAVE_PATH/train_or_test/str(label)
    current_save_path.mkdir(parents=True, exist_ok=True)
    np.save(current_save_path/patient_id, dcm_array)
    
    normalizer = 224 * 224
    if train_or_test == "train":
        sums += np.sum(dcm_array) / normalizer
        sums_squared += (np.power(dcm_array, 2).sum()) / normalizer
```


      0%|          | 0/26684 [00:00<?, ?it/s]



```python
# Defining and checking the mean and the standard deviation

mean = sums / len_train
stdev = np.sqrt(sums_squared / len_train - (mean**2))
print(f"Mean:\t\t\t {mean} \nStandard Deviation:\t {stdev}")
```

    Mean:			 0.44106992823128194 
    Standard Deviation:	 0.27758244159100576
    


```python
# Function to load data

def load_file(path):
    return(np.load(path).astype(np.float32))
```

The following cells perform random augmentations  on the dataset such as crops, rotations etc. to make the model more powerful in assessing low-quality images.


```python
# Data Augmentation Settings

train_transforms = transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean, stdev),
                                       transforms.RandomAffine(degrees     = (-5, 5),
                                                               translate   = (0, 0.05),
                                                               scale       = (0.9, 1.1)),
                                       transforms.RandomResizedCrop((224, 224),
                                                                    scale = (0.35, 1))
                                        ])

test_transforms = transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean, stdev),
                                        ])
```


```python
# Defining the train and test data

train = torchvision.datasets.DatasetFolder("Processed/train/",
                                           loader = load_file,
                                           extensions = "npy",
                                           transform = train_transforms)

test = torchvision.datasets.DatasetFolder("Processed/test/",
                                          loader = load_file,
                                          extensions = "npy",
                                          transform = test_transforms)
```


```python
# Viewing a random sample of 4

fig, axis = plt.subplots(2, 2, figsize = (9, 9))
for i in range(2):
    for j in range(2):
        random_index = np.random.randint(0, 24000)
        x_ray, label = train[random_index]
        axis[i][j].imshow(x_ray[0], cmap="bone")
        axis[i][j].set_title(f"Label:{label}")
```


    
![Figure 2](/assets/img/content/230303/output_25_0.png)  
<p style="text-align: center;"><em>Figure 2. Images After Augmentation</em></p>
    


The effect of augmentation can be seen clearly.


```python
batch_size = 64
num_workers = 2

train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, num_workers = num_workers, shuffle = True)
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, num_workers = num_workers, shuffle = False)

print(f"# of train images: \t{len(train)} \n# of test images: \t{len(test)}")
```

    # of train images: 	24000 
    # of test images: 	2684
    


```python
# Checking the number of images with and without pneumonia in the train set

np.unique(train.targets, return_counts = True)
```




    (array([0, 1]), array([18593,  5407]))



Since the data is imbalanced and the number of images without pneumonia are almost 3 times higher than images with pneumonia, a *weighted loss* of 3 can be used in the model, which means that the model will assign a higher penalty for the misclassification of the negative class.

## Modeling

We are going to use (ResNet-18)[https://arxiv.org/abs/1512.03385], which is an 18 layers deep convolutional neural network, along with the optimizer (Adam)[https://arxiv.org/abs/1412.6980]. The architecture of ResNet-18 can be checked with the following code:


```python
torchvision.models.resnet18()
```




    ResNet(
      (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      (layer1): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): BasicBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (layer2): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (downsample): Sequential(
            (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (layer3): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (downsample): Sequential(
            (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (layer4): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (downsample): Sequential(
            (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
      (fc): Linear(in_features=512, out_features=1000, bias=True)
    )




```python
class PneumoniaClassifier(pl.LightningModule):
    
    def __init__(self):

        super().__init__()
        
        self.model = torchvision.models.resnet18()
        # modifying the input channels of the first convolutional layer (conv1) from 3 to 1
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # modifying the out_features of the last fully connected layer (fc) from 1000 to 1
        self.model.fc = torch.nn.Linear(in_features=512, out_features=1, bias=True)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        # Adding the weigted loss to overcome the imbalance in the dataset
        self.loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3]))
        
        # Tracking the train and validation accuracy
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.valid_acc = torchmetrics.Accuracy(task="binary")

        # ---
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, data):
        # Computes the output of ResNet-18 and returns the prediction
        pred = self.model(data)
        return(pred)
    
    def training_step(self, batch, batch_idx):
        # PyTorch lightning optimizes according to the value returned by this function

        # Calculating the loss
        x_ray, label = batch
        label = label.float()
        pred = self(x_ray)[:,0]
        loss = self.loss_func(pred, label)

        # ---
        self.training_step_outputs.append(loss)
        
        # Recording accuracy
        self.log("Train Loss", loss)
        self.log("Step Train Acc", self.train_acc(torch.sigmoid(pred), label.int()))    # Converted to probability w/Sigmoid
        return(loss)
    
    def on_train_epoch_end(self):
        epoch_average = torch.stack(self.training_step_outputs).mean()
        self.log("training_epoch_average", epoch_average)
        self.training_step_outputs.clear()  # free memory
        
        
    def validation_step(self, batch, batch_idx):
        
        x_ray, label = batch
        label = label.float()
        pred = self(x_ray)[:,0]
        loss = self.loss_func(pred, label)
        
        # ---
        self.validation_step_outputs.append(loss)

        self.log("Validation Loss", loss)
        self.log("Step Validation Acc", self.valid_acc(torch.sigmoid(pred), label.int()))
        return(loss)
    
    def on_validation_epoch_end(self):
         epoch_average = torch.stack(self.validation_step_outputs).mean()
         self.log("validation_epoch_average", epoch_average)
         self.validation_step_outputs.clear()  # free memory
    
    def configure_optimizers(self):
        return([self.optimizer])
```


```python
model = PneumoniaClassifier()
```


```python
available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
available_gpus
```




    [<torch.cuda.device at 0x7f1ec6def6a0>]




```python
# Model checkpoint: Save top 10 checkpoints based on the highest validation accuracy

checkpoint_callback = ModelCheckpoint(monitor = 'Step Validation Acc',
                                      save_top_k = 10,
                                      mode = 'max')

# Creating the trainer

trainer = pl.Trainer(accelerator = "gpu", 
                     logger = TensorBoardLogger(save_dir= "./logs"), 
                     log_every_n_steps = 1,
                     callbacks = checkpoint_callback,
                     max_epochs = 40)
```

    INFO:pytorch_lightning.utilities.rank_zero:GPU available: True (cuda), used: True
    INFO:pytorch_lightning.utilities.rank_zero:TPU available: False, using: 0 TPU cores
    INFO:pytorch_lightning.utilities.rank_zero:IPU available: False, using: 0 IPUs
    INFO:pytorch_lightning.utilities.rank_zero:HPU available: False, using: 0 HPUs
    


```python
trainer.fit(model, train_loader, test_loader)
```

    WARNING:pytorch_lightning.loggers.tensorboard:Missing logger folder: ./logs/lightning_logs
    INFO:pytorch_lightning.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    INFO:pytorch_lightning.callbacks.model_summary:
      | Name      | Type              | Params
    ------------------------------------------------
    0 | model     | ResNet            | 11.2 M
    1 | loss_func | BCEWithLogitsLoss | 0     
    2 | train_acc | BinaryAccuracy    | 0     
    3 | valid_acc | BinaryAccuracy    | 0     
    ------------------------------------------------
    11.2 M    Trainable params
    0         Non-trainable params
    11.2 M    Total params
    44.683    Total estimated model params size (MB)


    INFO:pytorch_lightning.utilities.rank_zero:`Trainer.fit` stopped: `max_epochs=40` reached.
    


```python
# Set device to cuda if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.eval()
model.to(device)
```


```python
# Calculating predictions

preds = []
labels = []

with torch.no_grad():
    for data, label in tqdm(test):
        data = data.to(device).float().unsqueeze(0)
        
        pred = torch.sigmoid(model(data)[0].cpu())
        preds.append(pred)
        labels.append(label)
preds = torch.tensor(preds)
labels = torch.tensor(labels).int()
```


      0%|          | 0/2684 [00:00<?, ?it/s]



```python
# Checking the metrics

acc = torchmetrics.Accuracy(task="binary")(preds, labels)
precision = torchmetrics.Precision(task="binary")(preds, labels)
recall = torchmetrics.Recall(task="binary")(preds, labels)
cm = torchmetrics.ConfusionMatrix(task="binary")(preds, labels)

print(f"Accuracy:\t\t{acc}")
print(f"Precision:\t{precision}")
print(f"Recall:\t\t\t{recall}")
print(f"Confusion Matrix:\n {cm}")
```

    Accuracy:		0.8036512732505798
    Precision:	0.5432372689247131
    Recall:			0.8099173307418823
    Confusion Matrix:
     tensor([[1667,  412],
            [ 115,  490]])
    

High recall points out that the model rarely misses the cases with pneumonia, yet the precision is not sufficient and points out the high number of false positives.

*Written by Ahmet Yiğit Doğan*
[*Deep Learning with PyTorch for Medical Image Analysis*](https://www.udemy.com/course/deep-learning-with-pytorch-for-medical-image-analysis/)  
[GitHub Repository]()
