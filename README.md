# lateral-line
This project implements different neural network configurations for locating moving spheres under water in a simulated setting.


## Content of ths repo
At this moment this is part of a TensorFlow port of the original code in Theano. It currently implements the experiments 
using two parallel sensory arrays. The other part of this port from Theano to TF is yet to come.

## Running the code
The code consists of two main scripts: `generate_data.py` and `train.py`.
### Generating the data
First, to generate the data, you can do
```bash
python3 generate_data.py
```

There are several hyper parameters that you can pass, have a look at `latline/experiment_config.py`
to see which. To see a visualization of the simulation while generating it, you can do:
```bash
python3 generate_data.py --display
```

### Training the model
By running the script `train.py`
```bash
python3 train.py
```

You should see a model being trained. Also the script fully automatically chooses a log dir for
TensorBoard that you can use for immediate visualization of the training process.

