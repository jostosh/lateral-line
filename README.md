# lateral-line
This project implements different neural network configurations for locating moving spheres under water in a simulated setting.

## Simulated lateral line
Fish use their lateral line to locate nearby objects in water, allowing them to swim in schools, escape from predators or to detect pray. This organ senses water displacement. In our experiments we want to achieve detection of an arbitrary number of objects. Therefore, we formulate the prediction as a kind of density estimation. See [https://youtu.be/hSwGsQoFojM](this video) for an example in three dimensions. The paper will reveal how we can reconstruct the entire 3D density from just the measurements of two sensor arrays. All our expirements are conducted with synthesized data. 

## Content of ths repo
At this moment this is part of a TensorFlow port of the original code in Theano. It currently implements the experiments 
using two parallel sensory arrays.

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

