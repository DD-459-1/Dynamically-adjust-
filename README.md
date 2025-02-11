# Dynamically-adjust-
The code in this project demonstrates the performance of the Adam, Yogi, and AdaBelief optimizers with and without the DA/DAs strategies in long-term time series forecasting models.The time series prediction model implementation is adapted from the repository at https://github.com/thuml/Time-Series-Library. 


# Usage
1. Install Python 3.12 For convenience. execute the following command.

 ```bash
   pip install -r requirements.txt
```

2. Prepare Data. You can obtain the well pre-processed datasets from [Google Drive](https://drive.google.com/file/d/1xIuOoNFcvllsp3VQE5ZxulZqullEdB5L/view?usp=drive_link) or [Baidu Drive](https://pan.baidu.com/s/1hoW4dCq8SFGp0dCSD6kc-A?pwd=gea8). Then place the downloaded data in the folder `./dataset`
  
4. Simply execute the `run.py` file to initiate the training process. The dataset, model, optimizer, and other experimental parameters used during training can be configured within the `run.py` file.



