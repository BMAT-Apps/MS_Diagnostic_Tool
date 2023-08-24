# Multiple Sclerosis Diagnostic Tool 
This pipeline is a machine learning model based on a Logistic Regression to propose a binary diagnostic (MS or non-MS) based on advanced MRI biomarkers (Central Vein Sign (CVS), Cortical Lesion (CL), Paramagnetic Rim Lesion(PRL)). 

## Utilization
**Enter the information about the subject, Run Analysis and get the diagnosis !**

* Age: Enter the age of the subject in years as an int or float (e.g. 45.3)
* CVS: Enter the ratio of CVS of the subject. Use either on the following three rules:
  * Total CVS: enter the total ratio of CVS positive on CVS negative lesions as a percentage (e.g. 56%). This rule is yields better diagnostic but is time consuming to manually check.
  * Ro6 (Rule of 6): Enter the Rule of 6 results for CVS. Either Yes or No
  * Ro3 (Rule of 3): Enter the Rule of 3 results for CVS. Either Yes or No
  The model takes into account the most precise information available (Total CVS > Ro6 > Ro3).
* CL: Enter the number of CL of the subject
* PRL: Enter the number of PRL of the subject

**Run Diagnosis**

![MS Diagnostic Tool](/Readme_pictures/MSDT_window.gif)

## Stand-Alone installation
This tool can either be installed via the BMAT software or it can be downloaded and installed directly from this repository. For download and installation via BMAT, please refer to [BMAT](https://github.com/ColinVDB/BMAT).

The procedure for the stand-alone download and installation is desribed hereunder. F

### Depedencies

#### Python 
This software is written entirely in Python and thus requires Python to be installed to work properly. Pip is also needed to install the different packages that the software requires. Pip can sometimes be installed additionnaly when installing Python but not in every case, so it needs to be verified. The installation will be described for the different possible OS.

⚠️ **Python version should be >= 3.8** ⚠️

##### Linux

[Install Python on Linux](https://www.scaler.com/topics/python/install-python-on-linux/)

This link describes extensively how to install python on Linux. 

The easiest solution to install Python is to use the Package Manager. For this, open a terminal and type the following command: 

```
sudo apt-get install python
```

After the installation, you can verify that it all worked properly by typing:

```
python -V
```

This command should show you the version of Python. 

Then, you can check if pip has been installed by typing:

```
pip -V
```

If the command show you the version of pip, it means that it has been installed. Otherwise, it will say that 'pip' is not recognized and needs to be installed. To install pip, you can type in the terminal:

```
sudo apt-get install python-pip
```

This will install pip, you can check that pip is well installed by checking its version. 

##### Windows

The first possibility is to download Python via the microsoft store. This should download and install Python and Pip on your computer. To check if the installation has worked, you can open a command prompt or powershell and type:

```
python -V 
```

to see the version of python and 

```
pip -V
```

to see the verison of pip.

If there is no error, move on. 

The second possibility is to download the installation package directly on the [Python Website](https://www.python.org/downloads/windows/) and install it via the classic installation process. Following this method, pip will not be installed with python. Here are the steps to install pip:
1. Download the [get-pip.py](https://bootstrap.pypa.io/get-pip.py) script to a folder on your computer. 
2. Open a command prompt in that folder or navigate to that folder.
3. Run the script with the following command:

```
python get-pip.py
```

This should install pip. You can check if the installation worked by typing:

```
pip -V
```

##### Mac

Download the installation package directly from the [Python Website](https://www.python.org/downloads/macos/) and install it via the classic installation process. Following this method, pip will not be installed with python. Here are the steps to install pip:
1. Download the [get-pip.py](https://bootstrap.pypa.io/get-pip.py) script to a folder on your computer. 
2. Open a command prompt in that folder or navigate to that folder.
3. Run the script with the following command:

```
python get-pip.py
```

This should install pip. You can check if the installation worked by typing:

```
pip -V
```

### Installation

#### Download the application

This application can either be download via git, if you have git installed, via this command

```
git clone https://github.com/BMAT-Apps/MS_Diagnostic_Tool.git
```

*Git is free and open source software for distributed version control: tracking changes in any set of files, usually used for coordinating work among programmers collaboratively developing source code during software development. Please refer to [Git website](https://git-scm.com/downloads) for the intructions on how to install git on your specific OS*

The second possibility is to download the source code in zip file directly from this webpage by clicking on **Download Zip** in the *Code* drop-down menu of this page (see Figure below). Then you can extract the zip file into your perferred directory. 

![Download Zip file](/Readme_pictures/download_screenshot.png)

### Installing python package dependencies

To install all the required python package to run properly the application, open a terminal in this directory and run the following command:

```
pip install -r requirements.txt
```

### Launching application

To launch the application, open a terminal in the src folder of this repository and launch the file with this command:

```
python MS_diagnostic_tool.py
```




