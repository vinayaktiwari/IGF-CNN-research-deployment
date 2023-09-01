# IGF-CNN-research-deployment
Our work focused on outperforming or atleast matching the classification results of existing State-of-the-art pre-trained CNN models when dealing with Chest X-ray images. We proposed a simple feature extractor algorithm in conjunction with our custom three layered CNN architecture and ANN classifier. We call this approach Iterative Gaussian Feature Extractor with Custom CNN (IGF-CNN). We found that our approach not only matched and outperformed the existing works but also required significantly less parameters and training time. Our method gave out  99%, 99.85%, 99.8%, 98.8% and 96.75%  accuracy on five benchmarking datasets.                   


## Steps to run

### Step 1 - Clone the repo
```git@github.com:vinayaktiwari/IGF-CNN-research-deployment.git```
### Step 2 - Create virtual environment
```python3 -m venv venv_name```

```source venv_name/bin/activate```
### step 3 - Install requirements
``` pip install -r requirements.txt```

``` python3 app.py```

## Proposed Model Architecture




