# Setup and Installation

This project requires to have installed a few packages. Some of these packages need to be installed via conda, while others can be installed via pip.
Our suggestion is to create a new conda environment containing all the needed packages.

First, navigate to the project directory:

```bash
cd <project_directory>
```

From [requirements.txt](https://github.com/dslab-uniud/ppSTL-IJCAI2024/blob/main/supplementary_material/requirements.txt), prepare a file with packages available in conda and another for those only available in pip:

```bash
grep -v '=pypi_0' requirements.txt > conda_requirements.txt
grep '=pypi_0' requirements.txt | sed 's/=pypi_0//' > pip_requirements.txt
sed -i 's/=\+/==/g' pip_requirements.txt
```

To install the required packages, first use conda (creating a new local environment within your `<project_directory>`). Note that conda-forge channel might be required:

```bash
conda config --append channels conda-forge
conda create --prefix ./env --file conda_requirements.txt
```

Then, use pip for the remaining packages:

```bash
conda activate ./env
pip install -r pip_requirements.txt
```

Now, you should be able to run the scripts as described in the following "How to Execute the Code" section.




# How to Execute the Code
## Training mode

This project includes two main scripts: [framework_script_future.py](https://github.com/dslab-uniud/ppSTL-IJCAI2024/blob/main/supplementary_material/framework_code/framework_script_future.py) and [framework_script_past.py](https://github.com/dslab-uniud/ppSTL-IJCAI2024/blob/main/supplementary_material/framework_code/framework_script_past.py). They can be found in the [framework_code](https://github.com/dslab-uniud/ppSTL-IJCAI2024/tree/main/supplementary_material/framework_code) folder. To execute these scripts, you need to pass a dictionary containing all the necessary hyperparameters as a parameter.

Here's an example of how to run the scripts in training mode:

```bash
cd ./framework_code/
python3.9  framework_script_future.py input_params.txt
python3.9  framework_script_past.py input_params.txt
``` 

In the above commands, [input_params.txt](https://github.com/dslab-uniud/ppSTL-IJCAI2024/blob/main/supplementary_material/framework_code/input_params.txt) is a textual file containing a dictionary of hyperparameters. You can replace [input_params.txt](https://github.com/dslab-uniud/ppSTL-IJCAI2024/blob/main/supplementary_material/framework_code/input_params.txt) with the path to your own configuration file.

The output files of the script execution, generated in the [./runs/](https://github.com/dslab-uniud/ppSTL-IJCAI2024/tree/main/supplementary_material/framework_code/runs) folder, are several. Of general interest, there is a params file containing the configuration used to run the experiment (this file is needed for test purposes as well), and a pickle file containing the pool of formulas extracted with some statistics. 

Note that the script is currently configured to run in full training mode. To run it using a validation set (90-10 split), the lines at the end of the `main` function shall be changed (i.e., uncommented).





### Hyperparameters

The framework is highly flexible and allows for the setting of multiple hyperparameters. In this section, we describe the hyperparameters we considered for the hyperparameter search, with some default/example values (see also Appendix E, Table 2, in the [paper](https://github.com/dslab-uniud/ppSTL-IJCAI2024/blob/main/IJCAI_2024_framework_canonical.pdf)).

```text
'dataset_name': 'CMAPSS' - Name of the dataset. Based on it, train and test datasets are read (paths are hardcoded in the script) and also the corresponding preprocessing is applied
'min_accuracy': 0.75 - Parameter used to remove from the final genetic algorithm front the solutions which have an accuracy lower than that specified. Accuracy is evaluated on the original trace plus the augmented version of such a trace (only in case of a positive number of augmented traces, see next)
'n_augs': 50 - Number of times to augment the trace on which the genetic algorithm is executed
'max_score_ea': 0.02 - Score used as threshold within the genetic algorithm to filter out solutions that have a FAR greater than 'max_score_ea'.
'good_witnesses_ratio': 0.33 - Portion of good_witnessess traces to consider at each generation.
'max_gen': 500 - Maximum number of generations for each genetic algorithm execution
'pop_size': 200 - Size of the population
'cxpb': 0.8, - Crossover probability
'mutpb': 0.6 - Initial mutation probability
'mutation_decay': 3 - Decay parameter for mutation
'max_ea_height': 17 - Max height for each formula tree 
'goodbad_traces_at_ea_generation': 10 - If > 0, then good traces rotation is performed during EA generations. The number specifies after how many generations to sample new good traces. If 0, then no rotation is performed within the EA. Should always be >= 0. 
'ea_patience': 30 - After how many non-improving hypervolume generations to stop the EA
'output_file_suffix': 'future_s1' - String to insert in the output files name
'num_seeds': 1 - Specifies which init random seed is used among those pre-sampled.
```

To add a new dataset besides the one already provided ([CMAPSS](https://github.com/dslab-uniud/ppSTL-IJCAI2024/tree/main/supplementary_material/dataset/)), the script, particularly the `main` function, shall be edited to perform the necessary preprocessing/formatting steps.





## Test mode

To run the scripts in test mode, add the "test" parameter and ensure that the parameters dictionary points to the dictionary created by the script for the run of the pool that you want to use:

```bash
python3.9  framework_script_future.py test_params.txt test
python3.9  framework_script_past.py test_params.txt test
```

In this case, [test_params.txt](https://github.com/dslab-uniud/ppSTL-IJCAI2024/blob/main/supplementary_material/framework_code/test_params.txt) should contain the paths to the parameters file for the specific run of the pool of formulas that you want to test. 

###Example

Suppose that the execution of the framework led in the [./runs/](https://github.com/dslab-uniud/ppSTL-IJCAI2024/tree/main/supplementary_material/framework_code/runs) folder to the creation of the following two files:

```text
./runs/CMAPSS_future_s1_2024-01-16_19_53_30.395418_params
./runs/train_stl_results_CMAPSS_future_s1_2024-01-16_19_53_30.395418.pickle
```

The [test_params.txt](https://github.com/dslab-uniud/ppSTL-IJCAI2024/blob/main/supplementary_material/framework_code/test_params.txt) shall be done as follows:
```text
{
    'test_simulated_online' : False, 
    'train_dict_path': "./runs/CMAPSS_future_s1_2024-01-16_19_53_30.395418_params"
}
```



# Reproducing Our Experiments

To reproduce our experiments, you can use the provided configuration files which are located in the configs folder. Simply pass the desired configuration file as a parameter when running the scripts:

```bash
python3.9 framework_script_future.py configs/CMAPSS_future_s1.txt
python3.9 framework_script_past.py configs/CMAPSS_past_s1.txt
```

Replace [CMAPSS_future_s1.txt](https://github.com/dslab-uniud/ppSTL-IJCAI2024/blob/main/supplementary_material/framework_code/configs/CMAPSS_future_s1.txt) with the name of the configuration file for the experiment you want to reproduce.
