# Efficient Prompt Learning for Traffic Prediction #
This is the implementation of Simple Yet Effective Spatio-Temporal Prompt Learning in the following paper:

## Requirements ##
Pytorch = 1.12.0 with cuda 11.3

## Data ##
Point-based traffic data includes PeMSD03, PeMSD04, PeMSD07, PeMSD08 and PeMS-Bay. Grid-based traffic data includes NYCTaxi, TDrive and ChIBike.  Crime data includes data of New York City and Chicago.



## Hyperparameters ##
For fair comparison, all compared algorithms have hidden dimensionality modified from the range [8,16,32,64] to achieve their best performance as reported results at 32. The learning rate $\eta$ is initialized as 0.003 with weight decay 0.3. For GNN-based models, the number of GCN layer is 3. For prompt tuning network, the number of the TCN Layer is 2 and the number of MLP layer is set as 2. The kernel size of the TCN Layer is set as 7 during which our framework \model\ obtains the best performance from the range of [5,7,9,11]. Following existing settings of traffic prediction, we utilize historical 12 time steps with 5 minutes a step to predict future 12 time steps on point-based datasets (PeMSD04, PeMSD08, PeMSD03, PeMSD07 and PeMS-Bay). And we use historical 6 time steps to predict future 1 time step on grid-based datasets (NYCTaxi, CHIBike and TDrive). All baseline methods follow their predefined settings as their papers.  

## Simple Yet Effective Spatio-Temporal Prompt Learning Training and Predicting Process ##
    cd code_traffic
    Python Run.py -- Mode "train"   # Pretraining
    Python Run.py -- Mode "fine"   # Fine-tuning
    Python Run.py -- Mode "prompt"   # Prompt-Tuning
    cd code_crime
    Python Run.py -- Mode "train"   # Pretraining
    Python Run.py -- Mode "fine"   # Fine-tuning
    Python Run.py -- Mode "prompt"   # Prompt-Tuning










