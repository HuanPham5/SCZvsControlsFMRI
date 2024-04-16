# GAN-classification-fMRI

### Project advisor: Professor Majnu John, Professor Yihren Wu

### Student: Tieu Hoang Huan Pham

### Project description: Use WGAN for binary classification via transfer learning

## Getting Started
Install dependencies in requirements.txt

### Prerequisites
Install Anaconda

### Installation
* Make a virtual environment using Anaconda prompt
```sh
conda create --name gan python=3.11.5
```
* Activate the environment
```sh
conda activate gan
```
* Run this to install all dependencies
```sh
pip install -r requirements.txt
```
*To check how many environments you have:
```sh
conda info -e
```
or 
```sh
conda env list
```
To check if the environment is activated look at the:
```
(base) C:\Users\username>
```

## Usage
Inside the notebook, check the following:

* Under # GAN Training Data Selection, make sure the numbers are 50, 50, 9
* Change num_iterations
* Change directory path
* Change epochs

Then back to anaconda terminal
```sh
jupyter nbconvert --to python x-collapsed.ipynb
```
Then
```
python x-collapsed.ipynb >> output_log_x.txt
```




