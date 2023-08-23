# Template

This repository is a template to create a BMAT-Apps. 

## Requirement

Here is a list of mandatory required files and folders that need to be found in a BMAT-Apps:

* src/ folder: this folder contains the source code of the pipeline. The user must add its source code in python containing a dedicated graphical interface using PyQt5 and the computation code, as well as an associated JSON file containing metainformation about the pipeline. The python source code must contain a *launch* function to launch the graphical interface. The JSON file must be written in a dictionary-like structure and contain the following pieces of information:
   * "name”: the name of the Pipeline that will be displayed in the ‘Pipelines’ drop-down menu in the software.
   * “source_code”: the name of the python file containing the source code for the graphical interface of the pipeline. This file needs to be implemented using PyQt5 python module and contain a *launch* function, that takes the Main Window of the software and the "add_info" dictionary as arguments, and that launches the graphical interface of the Pipeline. User can take the Template pipeline as an example.
   * “import_name”: the name of the corresponding module that needs to be imported in python. It corresponds to the name of the python source code file without the ‘.py’ extension.
   * “attr”: the name of the attribute of the imported ‘Pipelines’ python module corresponding to the python source code of the pipeline. Again, this corresponds to the name of the python source code file without the ‘.py’ extension.
   * "add_info": this contains a dictionnary containing some specific information that the pipeline needs to run (e.g. the name of a sequence to use).

* *Readme.md* file: this file contains the documentation of the pipeline and explain how to use the pipeline with other useful information.

* *requirements.txt* file: this file contains all the python packages required to be installed for the pipeline to work properly.

* *setup.json* file: this file contains information for the BMAT software in order to be able to install the pipeline properly. More specifically, this JSON file must be written in a dictionary-like structure and contain the following pieces of information:
   * "python_requirements": contains the name of the *requirements* file containing a list of every required python package that needs to be installed for the pipeline to work properly ("requirements.txt" by default).
   * "docker": this key contains a dictionary with information about the installation of the docker container used by the pipeline (if the pipeline uses a docker container, otherwise the keys of this container is put to null). The dictionary must contain these keys:
       * "docker": name of the docker image to pull | "Dockerfile" to build the image from a Dockerfile (in that case, a Dockerfile is needed in the repository) | *null* if the pipeline does not use a docker image.
       * "tag": new name of the docker image if the user wants to rename the docker image that he pulled (this is required with "Dockerfile" option to name the built docker container)

The easiest way to implement a new BMAT-App is to fork this Template BMAT-App which already contains the right structure.


