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


