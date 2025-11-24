### 1. Environment
* Ubuntu 20.04
* Nvidia RTX 3090 Ti 24G
* cuda 12.1
* python 3.10
* torch 2.3.1

For reference only.

### 2. Installation
Clone the repository
```bash
git clone https://github.com/loztsky/STAR-Net.git
```
Create environment
```bash
conda create -n starnet python=3.10
```
Install dependencies
```bash
pip install torch==2.3.1 torchvision==0.18.1
```
Install dependencies
```bash
pip install -r requirements.txt
```
Install as local library
```bash
pip install -e .
```