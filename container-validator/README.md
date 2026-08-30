# FOMO26 Container Validator

For instructions on
1. how to prepare your code for submission, please see [Preparing your model for submission](https://www.synapse.org/Synapse:syn72120565/wiki/640840).
2. how to submit your container, please see [Submission instructions](https://www.synapse.org/Synapse:syn72120565/wiki/640841).

## Installation

1. Install Apptainer
You need to install Apptainer (formerly Singularity) to build and run your container. Installation instructions by platform:
- [Install in Linux (Ubuntu, Debian, Fedora, ...)](https://apptainer.org/docs/admin/main/installation.html#install-from-pre-built-packages)
- [Install in MacOS](https://apptainer.org/docs/admin/main/installation.html#mac)
- [Install in Windows](https://apptainer.org/docs/admin/main/installation.html#windows)

If using MacOS and Windows please follow this guide from _within_ your virtual environment.

Once you have installed it, verify your Apptainer installation with:

```bash
apptainer --version
```

## Build image

```bash
apptainer build --fakeroot /path/to/save/your/container.sif path/to/Apptainer.def --arch amd64
```


## How To Run Validation

To run the local validator first install the required packages (in the distro or VM where apptainer is also executed)
```
pip install -r requirements.txt
```

Use the following command and replace TASK with one of the valid tasks: `task1`, `task2`, `task3`, `task4`, `task5`, and `task6_and_7` and the dummy path with the path to your container

```
python3 container_validator/validate.py --task TASK --sif PATH/TO/YOUR/CONTAINER.sif 
```

relevant flags:
`--no-gpu` to test without requiring local GPU libraries.

Once your container is valid you will see the following output (where X is the number of tests we run for that task):
```
================================================================
  ALL X TESTS PASSED — container is ready to submit!
================================================================
```
