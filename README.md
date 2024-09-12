Code for the GT tau performance enhancer

for usage in the cern swan infrastructure.


1) Setup:

 Update the python ```awkward ``` library:
 1) Inside swan, open a terminal and run:
  ```
pip install --user awkward --upgrade
pip install --user pyarrow --upgrade
pip install --user vector
```
 2) Restart swan afterwards with the following configuration:


![image](https://github.com/user-attachments/assets/d326aafa-f626-4928-8b43-d82fee060b37)

It is recommended to request access to a GPU via the cern service portal as 32 GB of ram are required for the minimum bias evaluation.

 3) Clone this repository in swan via the ```git > Clone a repository  ``` option in swan.
 4) Enter the tau-pe directory and 
 5) run the ```setup.ipynb ``` notebook, which clones the Menu Tools and generates  required input data (might take a few hours, but has to be done only once)



The notebooks are intended to be run in the following order:
1) ```smallnetdata.ipynb```
2) ```smallnet_modeltraining.ipynb```
3) ```rte.ipynb``` To get the rate plots this file has to be run with ``` sample = "MinBias" ```


to evaluate the performance and produce the plots, run:

```
cd tau-pe
unset PYTHONHOME && unset PYTHONPATH && cd Phase2-L1MenuTools/
. menuenv/bin/activate
cp ../taus.yaml configs/V37nano/objects/taus.yaml
cp ../taus_rate.yaml configs/V37nano/rate_plots/
rate_plots configs/V37nano/rate_plots/taus_rate.yaml 
cp ../gttau_matching.yaml configs/V37nano/object_performance/
object_performance configs/V37nano/object_performance/gttau_matching.yaml 
```
the Plots are in:
```
tau-pe/Phase2-L1MenuTools/outputs/V37nano/object_performance/
```

 
