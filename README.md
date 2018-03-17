# tensorflow-wrapper
Sklearn style wrapper around tensorflow models, where the configuration parameters can be set up using json file.

Usage guide:
Step 1: Define the computational graph (tensorflow model) in the "models/" folder, by inheriting the models.scg.SCG base class. Check out the existing models for examples

Step 2: Define the configuration file that your model needs inside "configs/" folder as a json file. There are some mandatory fields in json configuration file such as - "datadir", "logdir" which are applicable for the framework itself. Go through existing config files as an example

Step 3: Define a pre processing function for your data type. Functions to pre-process certain data types is included in framework.pre_process, which can be reused if it suits your needs.

Step 4: import the model, pre processing function and instantiate an object of framework.classifier.Classifier passing the config file location as argument. Check out the provided example iPython notebook "example_usage.ipynb" for usage guidelines.
