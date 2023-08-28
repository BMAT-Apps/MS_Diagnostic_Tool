#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 14:25:40 2021

@author: ColinVDB
Template
"""


import sys
import os
from os.path import join as pjoin
from os.path import exists as pexists
# from dicom2bids import *
import logging
from PyQt5.QtCore import (QSize,
                          Qt,
                          QModelIndex,
                          QMutex,
                          QObject,
                          QThread,
                          pyqtSignal,
                          QRunnable,
                          QThreadPool)
from PyQt5.QtWidgets import (QDesktopWidget,
                             QApplication,
                             QWidget,
                             QPushButton,
                             QMainWindow,
                             QLabel,
                             QLineEdit,
                             QVBoxLayout,
                             QHBoxLayout,
                             QFileDialog,
                             QDialog,
                             QTreeView,
                             QFileSystemModel,
                             QGridLayout,
                             QPlainTextEdit,
                             QMessageBox,
                             QListWidget,
                             QTableWidget,
                             QTableWidgetItem,
                             QMenu,
                             QAction,
                             QTabWidget,
                             QCheckBox, 
                             QRadioButton)
from PyQt5.QtGui import (QFont,
                         QIcon)

# specific import
import json
import numpy as np 
from sklearn.linear_model import LogisticRegression
import pandas as pd



def launch(parent, add_info=None):
    """
    

    Parameters
    ----------
    parent : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
        
    pwd = pjoin(__file__.replace('MS_diagnostic_tool.py', ''))
    window = MainWindow(parent, add_info={'pwd':pwd})
    window.show()



# =============================================================================
# MainWindow
# =============================================================================
class MainWindow(QMainWindow):
    """
    """
    

    def __init__(self, parent, add_info):
        """
        

        Parameters
        ----------
        parent : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        super().__init__()
        self.parent = parent
        # self.bids = self.parent.bids
        self.add_info = add_info
        self.pwd = add_info['pwd']

        self.setWindowTitle("Multiple Sclerosis Diagnostic Tool")
        self.window = QWidget(self)
        self.setCentralWidget(self.window)
        self.center()
        
        self.tab = MSDT_Tab(self)
        layout = QVBoxLayout()
        layout.addWidget(self.tab)

        self.window.setLayout(layout)


    def center(self):
        """
        

        Returns
        -------
        None.

        """
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())



# =============================================================================
# TemplateTab
# =============================================================================
class MSDT_Tab(QWidget):
    """
    """
    

    def __init__(self, parent):
        """
        

        Parameters
        ----------
        parent : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        super().__init__()
        self.parent = parent
        self.pwd = parent.pwd
        # self.bids = self.parent.bids
        self.setMinimumSize(500, 200)
        
        presentation_message = """
Multiple Sclerosis Diagnosis Tool:
    Enter the information of the following biomarkers for each subject and 
    see if the subject is dignosed with MS or not (with a confidence level)
    Biomarkers:
        CVS (Central Vein Sign): Either % of CVS (preferred)
                                 Ro6 (Rule of 6): Yes/No
                                 Ro3 (Rule of 3): Yes/No
        CL (Cortical Lesions): Number of CL
        PRL (Paramgnetic Rim Lesions): Number of PRL
        """
        
        self.label = QLabel()
        
        # Age
        self.age_lab = QLabel('Age:')
        self.age_input = QLineEdit(self)
        self.age_input.setPlaceholderText('Age (years)')
        
        # CVS
        self.cvs_lab = QLabel('CVS:')
        self.cvs_widget = CVSWidget(self)
        
        # CL
        self.cl_lab = QLabel('CL:')
        self.cl_input = QLineEdit(self)
        self.cl_input.setPlaceholderText('Number of CL')
        
        # PRL
        self.prl_lab = QLabel('PRL:')
        self.prl_input = QLineEdit(self)
        self.prl_input.setPlaceholderText('Number of PRL')
        
        # Results
        self.diagnosis_lab = QLabel('Diagnosis:')
        self.result = QLabel(self)
        
        # button
        self.button = QPushButton("Run Diagnosis")
        self.button.clicked.connect(self.run_diagnosis)
        
        layout = QGridLayout()
        layout.addWidget(self.label, 0, 0, 1, 2)
        layout.addWidget(self.age_lab, 1, 0)
        layout.addWidget(self.age_input, 1, 1)
        layout.addWidget(self.cvs_lab, 2, 0)
        layout.addWidget(self.cvs_widget, 2, 1)
        layout.addWidget(self.cl_lab, 3, 0)
        layout.addWidget(self.cl_input, 3, 1)
        layout.addWidget(self.prl_lab, 4, 0)
        layout.addWidget(self.prl_input, 4, 1)
        layout.addWidget(self.diagnosis_lab, 5, 0)
        layout.addWidget(self.result, 6, 0, 1, 2)
        layout.addWidget(self.button, 7, 0, 1, 2)
        layout.addWidget(self.button)
        self.setLayout(layout)


    def run_diagnosis(self):
        """
        

        Returns
        -------
        None.

        """
        age = None
        try:
            age = float(self.age_input.text())
        except ValueError as e:
            pass
        cvs_infos = self.cvs_widget.get_cvs_info()
        cl = None
        try:
            cl = int(self.cl_input.text())
        except ValueError as e:
            pass
        prl = None
        try:
            prl = int(self.prl_input.text())
        except ValueError as e:
            pass
        
        # check of information
        if age == None:
            print('Age information is mandatory')
            diagnosis_result = 'Age information is mandatory'
        elif cvs_infos['cvs'] == None and cvs_infos['ro6'] == None and cvs_infos['ro3'] == None:
            print('CVS information is mandatory')
            diagnosis_result = 'CVS information is mandatory'
        
        elif cl == None:
            print('CL information is mandatory')
            diagnosis_result = 'CL information is mandatory'
        
        elif prl == None:
            print('PRL information is mandatory')
            diagnosis_result = 'PRL information is mandatory'
            
        else:
            diagnosis_result = self.run_diagnostic(age, cvs_infos, cl, prl)
        
        # print result dignosis
        self.result.setText(diagnosis_result)
        
        
    def run_diagnostic(self, age, cvs_infos, cl, prl):
        print('bruh ?')
        # Base don infos, select right model
        model_name = None
        X_test = None
        if cvs_infos['cvs'] != None:
            model_name = 'CVS_CL_PRL_model'
            X_test = pd.DataFrame({'age':age, 'CVS':[cvs_infos['cvs']], 'nCL':[cl], 'nPRL':[prl]})
        elif cvs_infos['ro6'] != None:
            model_name = 'Ro6_CL_PRL_model'
            X_test = pd.DataFrame({'age':age, 'CVS':[cvs_infos['ro6']], 'nCL':[cl], 'nPRL':[prl]})
        elif cvs_infos['ro3'] != None:
            model_name = 'Ro3_CL_PRL_model'
            X_test = pd.DataFrame({'age':age, 'CVS':[cvs_infos['ro3']], 'nCL':[cl], 'nPRL':[prl]})
        else:
            return 'Model to use not recognized'
        
        print(model_name)
        ## import model
        with open(pjoin(self.pwd, 'models', f'{model_name}.json'), 'r') as f:
            model_json = json.load(f)
        
        #instantiate the model
        X = pd.DataFrame(model_json['X'])
        y = pd.DataFrame(model_json['y'])
        model = LogisticRegression()
        model.fit(X, y)

        #fit the model using the training data
        coef = np.array(model_json['coef'])
        intercept = np.array(model_json['intercept'])
        model.coef_ = coef
        model.intercept_ = intercept
        
        pred = model.predict(X_test)
        pred = pred[0]
        pred_proba = model.predict_proba(X_test)
        pred_proba = pred_proba[::,pred]
        conf_lvl = pred_proba[0]
        
        diagnostic = 'Multiple Sclerosis' if pred == 1 else 'Non Multiple Sclerosis'
        
        return f'{diagnostic} \t (Likelihood = %.4f)' % conf_lvl
        
    
    def close_app(self):
        pass
    
    
# =============================================================================
# CVS Widget
# =============================================================================
class CVSWidget(QWidget):
    """
    """
    

    def __init__(self, parent):
        """
        

        Parameters
        ----------
        parent : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        super().__init__()
        self.parent = parent
        # self.bids = self.parent.bids
        # self.setMinimumSize(500, 200)
        
        # CVS
        self.cvs_tot_lab = QLabel('Total CVS:')
        self.cvs_tot_input = QLineEdit(self)
        self.cvs_tot_input.setPlaceholderText('Total CVS (e.g. 65)')
        self.cvs_tot_p = QLabel('%')
        
        self.ro6_lab = QLabel('Select6*:')
        self.ro6_y = QCheckBox('Yes')
        self.ro6_n = QCheckBox('No')
        self.ro6_y.stateChanged.connect(self.ro6_y_toggle)
        self.ro6_n.stateChanged.connect(self.ro6_n_toggle)
        
        self.ro3_lab = QLabel('Select3*:')
        self.ro3_y = QCheckBox('Yes')
        self.ro3_n = QCheckBox('No')   
        self.ro3_y.stateChanged.connect(self.ro3_y_toggle)
        self.ro3_n.stateChanged.connect(self.ro3_n_toggle)
        
        layout = QGridLayout()
        layout.addWidget(self.cvs_tot_lab, 0, 0)
        layout.addWidget(self.cvs_tot_input, 0, 1, 1, 2)
        layout.addWidget(self.cvs_tot_p, 0, 3)
        layout.addWidget(self.ro6_lab, 1, 0)
        layout.addWidget(self.ro6_y, 1, 1)
        layout.addWidget(self.ro6_n, 1, 2)
        layout.addWidget(self.ro3_lab, 2, 0)
        layout.addWidget(self.ro3_y, 2, 1)
        layout.addWidget(self.ro3_n, 2, 2)
        # layout.addWidget(self.button)
        self.setLayout(layout)

    def ro6_y_toggle(self):
        if self.ro6_y.isChecked():
            if self.ro6_n.isChecked():
                self.ro6_n.setChecked(False)
    def ro6_n_toggle(self):
        if self.ro6_n.isChecked():
            if self.ro6_y.isChecked():
                self.ro6_y.setChecked(False)
    def ro3_y_toggle(self):
        if self.ro3_y.isChecked():
            if self.ro3_n.isChecked():
                self.ro3_n.setChecked(False)
    def ro3_n_toggle(self):
        if self.ro3_n.isChecked():
            if self.ro3_y.isChecked():
                self.ro3_y.setChecked(False)


    def get_cvs_info(self):
        cvs_tot = None
        try:
            cvs_tot = float(self.cvs_tot_input.text())
        except ValueError as e:
            pass
        ro6 = None
        if self.ro6_y.isChecked():
            ro6 = True
        if self.ro6_n.isChecked():
            ro6 = False
        ro3 = None
        if self.ro3_y.isChecked():
            ro3 = True
        if self.ro3_n.isChecked():
            ro3 = False
        infos = {'cvs':cvs_tot, 'ro6':ro6, 'ro3':ro3}
        return infos
        


# # =============================================================================
# # ActionWorker
# # =============================================================================
# class ActionWorker(QObject):
#     """
#     """
#     finished = pyqtSignal()
#     progress = pyqtSignal(int)
    

#     def __init__(self):
#         """
        

#         Returns
#         -------
#         None.

#         """
#         super().__init__()
        

#     def run(self):
#         """
        

#         Returns
#         -------
#         None.

#         """
#         # Action
#         print('Beginning of the action')
#         print('End of the action')
#         self.finished.emit()
        
        
if __name__ == '__main__':
    
    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()
    
    launch(None, add_info=None)
    
    app.exec()


