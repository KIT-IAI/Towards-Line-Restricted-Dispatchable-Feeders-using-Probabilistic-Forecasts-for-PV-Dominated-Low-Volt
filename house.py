#################################### Packages ####################################
import json
import pickle
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.pyplot import figure
from casadi import *
import copy


####################################Class House####################################
class House:
    def __init__(self, parameter, start_compute, end_compute):
        #Define results
        #Time of computation
        self.start_compute = start_compute
        self.end_compute = end_compute
        self.hours = pd.date_range(start=start_compute, end=end_compute, freq='H')
        self.days = np.unique(self.hours.day)
        #Time of schedule
        self.start_schedule = start_compute + dt.timedelta(hours=parameter['len_compute'])
        self.end_schedule = end_compute + dt.timedelta(hours=parameter['len_DS_extension'])
        self.index_DS = pd.date_range(start=self.start_schedule, end=self.end_schedule, freq='H')

    #Write decision variables into list
    def dec_var_to_list(self, stage):
        liste = []
        for var in self.dec_var(stage):
            liste.append(self.__dict__[var])
        return(liste)

    #Split list and write into decision variables
    def list_to_dec_var(self, stage, liste):
        it = iter(liste)
        for var in self.dec_var(stage):
            self.__dict__[var] = next(it)
    
    #Write optimal decision variables into results
    def dec_var_to_results(self, stage, index, length):
        for var in self.dec_var(stage):
            if var == 'e' and stage != 'stage1' and not isinstance(self, PerfectHouse):
                self.__dict__[stage + '_' + var]['value'][index+1:index+1+length] = self.__dict__[var][1:1+length]
            else:
                self.__dict__[stage + '_' + var]['value'][index:index+length] = self.__dict__[var][0:length]


####################################Subclass Probabilistic House####################################    
class ProbabilisticHouse(House):
    def __init__(self, parameter, start_compute, end_compute):
        #Define decision variables
        self.g = None
        self.g_plus = None 
        self.g_minus = None
        self.delta_g = None
        self.p = None
        self.p_plus = None
        self.p_minus = None
        self.e = None
        self.epsilon = None
        self.rho = None
        
        #Inheritance Superclass
        super().__init__(parameter, start_compute, end_compute)
        #Initialize results
        blank_shape = np.rec.fromarrays((self.index_DS, np.zeros(len(self.index_DS))), names=('time','value'))
        self.estimate_e = np.rec.fromarrays((self.days[1:], np.zeros(len(self.days[1:]))), names=('day','value'))
        self.stage1_forecast_power = blank_shape.copy()
        self.stage1_forecast_power_lb = blank_shape.copy()
        self.stage1_forecast_power_ub = blank_shape.copy()
        self.stage2_forecast_power = blank_shape.copy()
        self.actual_power = blank_shape.copy()
        for stage in ['stage1', 'stage2', 'actual']:
            for var in self.dec_var(stage):
                self.__dict__[stage + '_' + var] =  blank_shape.copy()
        
    #Define decision variables for different stages
    def dec_var(self, stage):
        if stage == 'stage1':
            keys = ['g', 'g_plus', 'g_minus', 'p', 'p_plus', 'p_minus', 'e', 'rho', 'epsilon']
        elif stage == 'stage2':
            keys = ['g', 'p', 'p_plus', 'p_minus', 'e']
        elif stage == 'actual':
            keys = ['g', 'delta_g', 'p', 'p_plus', 'p_minus', 'e']
        else:
            print('Wrong stage description')
        return(keys)


####################################Subclass Perfect House####################################
class PerfectHouse(House):
    def __init__(self, parameter, start_compute, end_compute):
        #Define decision variables
        self.g = None
        self.g_plus = None 
        self.g_minus = None
        self.p = None
        self.p_plus = None
        self.p_minus = None
        self.e = None
        
        #Inheritance Superclass
        super().__init__(parameter, start_compute, end_compute)
        #Initialize results
        blank_shape = np.rec.fromarrays((self.index_DS, np.zeros(len(self.index_DS))), names=('time','value'))
        self.actual_power = blank_shape.copy()
        stage = 'actual'
        for var in self.dec_var(stage):
            self.__dict__[stage + '_' + var] =  blank_shape.copy()
        
    #Define decision variables for different stages
    def dec_var(self, stage):
        keys = ['g', 'g_plus', 'g_minus', 'p', 'p_plus', 'p_minus', 'e']
        return(keys)
    
####################################Subclass Deterministic House####################################
class DeterministicHouse(House):
    def __init__(self, parameter, start_compute, end_compute):
        #Define decision variables
        self.g = None
        self.g_plus = None 
        self.g_minus = None
        self.delta_g = None
        self.p = None
        self.p_plus = None
        self.p_minus = None
        self.e = None
        
        #Inheritance Superclass
        super().__init__(parameter, start_compute, end_compute)
        #Initialize results
        blank_shape = np.rec.fromarrays((self.index_DS, np.zeros(len(self.index_DS))), names=('time','value'))
        self.estimate_e = np.rec.fromarrays((self.days[1:], np.zeros(len(self.days[1:]))), names=('day','value'))
        self.stage1_forecast_power = blank_shape.copy()
        self.stage2_forecast_power = blank_shape.copy()
        self.actual_power = blank_shape.copy()
        for stage in ['stage1', 'stage2', 'actual']:
            for var in self.dec_var(stage):
                self.__dict__[stage + '_' + var] =  blank_shape.copy()
        
    #Define decision variables for different stages
    def dec_var(self, stage):
        if stage == 'stage1':
            keys = ['g', 'g_plus', 'g_minus', 'p', 'p_plus', 'p_minus', 'e']
        elif stage == 'stage2':
            keys = ['g', 'p', 'p_plus', 'p_minus', 'e']
        elif stage == 'actual':
            keys = ['g', 'delta_g', 'p', 'p_plus', 'p_minus', 'e']
        else:
            print('Wrong stage description')
        return(keys)