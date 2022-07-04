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


####################################2. Stage: MPC####################################
def mpc(parameter, houses, forecast_median, actual_e, index_time):
    
    #Define variables
    objective = 0
    #Initialize constraints
    constraints_list = []
    lb_list = []
    ub_list = []
    #Initialize list of all decision variables
    decision = []
    
    for house in houses:
        house.g = SX.sym('g', parameter['len_MPC'])
        house.p = SX.sym('p', parameter['len_MPC'])
        house.p_plus = SX.sym('p+', parameter['len_MPC'])
        house.p_minus = SX.sym('p-', parameter['len_MPC'])
        house.e = SX.sym('e', parameter['len_MPC']+1)
        #Extend list of all decision variables
        decision.extend(house.dec_var_to_list('stage2'))
        #print('DS', house.stage1_g['value'][index_time:index_time+parameter['len_MPC']])
    
        for j in range(parameter['len_MPC']):
            objective = objective + (house.stage1_g['value'][index_time+j] - house.g[j])**2 
        
    #Add remaining constraints via define_constraints()    
    #(houses, constraints_list, lb_list, ub_list, length, forecast_median, e) 
    define_constraints(houses, constraints_list, lb_list, ub_list, parameter['len_MPC'], forecast_median, \
                       actual_e)

    #Initialize optimization problem
    nlp = {}
    nlp['x'] = vertcat(*decision)
    nlp['f'] = objective
    nlp['g'] = np.asarray(constraints_list)
    lower_bound = np.asarray(lb_list)
    upper_bound = np.asarray(ub_list)
    
    #Solve optimization problem
    opts = {'ipopt.print_level':0, 'print_time':0, 'ipopt.max_iter':10000000, 'ipopt.tol': 1e-10}
    F = nlpsol('F','ipopt',nlp,opts)
    opti = F(x0 = np.zeros(vertcat(*decision).shape[0]), lbg = lower_bound, ubg = upper_bound)
    opti_decision_list = np.squeeze(opti['x'])
    
    #Unravel results
    it1 = iter(opti_decision_list) 
    sizes = [item.shape[0] for item in decision]
    it2 = iter([[next(it1) for _ in range(size)] for size in sizes])                 
    for house in houses:
        liste = [next(it2) for _ in range(len(house.dec_var('stage2')))]                
        house.list_to_dec_var('stage2', liste)  
    
    assert F.stats()['return_status'] == 'Solve_Succeeded', "Assertion in 2stage_MPC:" + F.stats()['return_status']
    return(np.squeeze(opti['f']))
        
    
####################################3. Stage: MPC####################################
def controlled_ESS(parameter, houses, actual_power, actual_e, DS, stage3 = True):
    
    #Define variables
    objective = 0
    #Initialize constraints
    constraints_list = []
    lb_list = []
    ub_list = []
    #Initialize list of all decision variables
    decision = []
    
    for i, house in enumerate(houses):
        house.g = SX.sym('g', 1)
        house.delta_g = SX.sym('delta_g', 1)
        house.p = SX.sym('p', 1)
        house.p_plus = SX.sym('p+', 1)
        house.p_minus = SX.sym('p-', 1)
        house.e = SX.sym('e', 2)
        #Extend list of all decision variables
        decision.extend(house.dec_var_to_list('actual'))
    
        objective = objective + house.delta_g**2
    
        #g = g_ref + delta_g
        constraint_actual_g = house.g - DS['house' + str(i)] - house.delta_g
        lb_actual_g = 0
        ub_actual_g = 0
        constraints_list.append(constraint_actual_g)
        lb_list.append(lb_actual_g)
        ub_list.append(ub_actual_g)
    
    
    #(houses, constraints_list, lb_list, ub_list, length, forecast_median, e)
    define_constraints(houses, constraints_list, lb_list, ub_list, 1, actual_power, actual_e)
 
    #Initialize optimization problem
    nlp = {}
    nlp['x'] = vertcat(*decision)
    nlp['f'] = objective
    nlp['g'] = np.asarray(constraints_list)
    lower_bound = np.asarray(lb_list)
    upper_bound = np.asarray(ub_list)
    
    #Save optimization problem
    opts = {'ipopt.print_level':0, 'print_time':0, 'ipopt.max_iter':10000000, 'ipopt.tol': 1e-11}
    F = nlpsol('F','ipopt',nlp,opts)
    opti = F(x0 = np.zeros(vertcat(*decision).shape[0]), lbg = lower_bound, ubg = upper_bound)
    opti_decision_list = np.squeeze(opti['x'])
    
    #Unravel results
    it1 = iter(opti_decision_list) 
    sizes = [item.shape[0] for item in decision]
    it2 = iter([[next(it1) for _ in range(size)] for size in sizes])                 
    for house in houses:
        liste = [next(it2) for _ in range(len(house.dec_var('actual')))]                
        house.list_to_dec_var('actual', liste)  
    
    if stage3 == True:
        assert F.stats()['return_status'] == 'Solve_Succeeded', \
        "Assertion in 3stage_ControlledESS or EstimateSOE:" + F.stats()['return_status']
    return(np.squeeze(opti['f']))


####################################Joint Constraints for all Stages####################################
def define_constraints(houses, constraints_list, lb_list, ub_list, length, forecast_median, e):
   
    for k in range(0,length):  
        for i, house in enumerate(houses): 
            
            #e(k+1) = e(k) + delta*(p(k) - mu*p+(k) + mu*p-(k))
            constraint_SOE_balance = house.e[k+1] - house.e[k] - parameter['delta']*(house.p[k] - \
                                     parameter['mu']*house.p_plus[k] + parameter['mu']*house.p_minus[k])
            lb_SOE_balance = 0
            ub_SOE_balance = 0
            constraints_list.append(constraint_SOE_balance)
            lb_list.append(lb_SOE_balance)
            ub_list.append(ub_SOE_balance)

            #g(k) = p(k) + l(k) 
            constraint_power_exchange = house.g[k] - house.p[k] - forecast_median['house' + str(i)][k] 
            lb_power_exchange = 0
            ub_power_exchange = 0
            constraints_list.append(constraint_power_exchange)
            lb_list.append(lb_power_exchange)
            ub_list.append(ub_power_exchange)

            #p(k) = p+(k) + p-(k)
            constraint_split_power = house.p[k] - house.p_plus[k] - house.p_minus[k]
            lb_split_power = 0 
            ub_split_power = 0
            constraints_list.append(constraint_split_power)
            lb_list.append(lb_split_power)
            ub_list.append(ub_split_power)

            #p+(k) >= 0 
            constraint_p_plus = house.p_plus[k]
            lb_p_plus = 0
            ub_p_plus = house.p_max
            constraints_list.append(constraint_p_plus)
            lb_list.append(lb_p_plus)
            ub_list.append(ub_p_plus)

            #p-(k) <= 0 
            constraint_p_minus = house.p_minus[k]
            lb_p_minus = house.p_min
            ub_p_minus = 0
            constraints_list.append(constraint_p_minus)
            lb_list.append(lb_p_minus)
            ub_list.append(ub_p_minus)

            #p+(k)*p-(k) <= relaxation_p
            constraint_complement_power = house.p_plus[k]*(-house.p_minus[k])
            lb_complement_power = 0
            ub_complement_power = parameter['relaxation_p']
            constraints_list.append(constraint_complement_power)
            lb_list.append(lb_complement_power)
            ub_list.append(ub_complement_power)

            #e_min <= e(k+1) <= e_max
            constraint_SOE_bounds = house.e[k+1]
            lb_SOE_bounds = house.e_min
            ub_SOE_bounds = house.e_max
            constraints_list.append(constraint_SOE_bounds)
            lb_list.append(lb_SOE_bounds)
            ub_list.append(ub_SOE_bounds)
            
            #e(kb) = e
            if k == 0:
                constraint_e_start = house.e[k]
                lb_e_start = e['house' + str(i)]
                ub_e_start = e['house' + str(i)]
                constraints_list.append(constraint_e_start)
                lb_list.append(lb_e_start)
                ub_list.append(ub_e_start)

        #-g_max <= sum(g(k)) <= g_max
        constraint_restrict_sum_g = sum(house.g[k] for house in houses)
        lb_restrict_sum_g = -parameter['g_max']
        ub_restrict_sum_g = parameter['g_max']
        constraints_list.append(constraint_restrict_sum_g)
        lb_list.append(lb_restrict_sum_g)
        ub_list.append(ub_restrict_sum_g)
        
        
####################################Estimate SoE####################################        
def estimate_SOE(parameter, houses, forecast_median, e, DS):
    e_temp = copy.deepcopy(e)
    for k2 in range(parameter['len_compute']):
        forecast_median_extract = {keys: values[k2:] for keys, values in forecast_median.items()}
        DS_extract = {keys: values[k2] for keys, values in DS.items()}
        #(parameter, houses, actual_power, actual_e, DS)
        controlled_ESS(parameter, houses, forecast_median_extract, e_temp, DS_extract, False)
        for i, house in enumerate(houses):
            e_temp['house' + str(i)] = house.e[1]
    return(e_temp)


####################################Calculate Forecast Quantities based on Quantiles####################################
def calculate_forecast_quantities(q_power, q_energy, hour, day = False):
    #Attention: len(q_power) and len(q_energy) differ
    forecast_median = np.zeros(len(q_power))
    
    #Median for Power
    forecast_median = q_power[:,49,hour]
 
    if not isinstance(day, bool):
        assert(hour / 24 == day)
        forecast_probabilistic = {}
        forecast_probabilistic['lower_bound'] = np.zeros(len(q_power)-parameter['len_compute'])
        forecast_probabilistic['upper_bound'] = np.zeros(len(q_power)-parameter['len_compute'])
        forecast_probabilistic['CDF'] = [None]*(len(q_energy)-parameter['len_compute'])
        
        #Lower Bound for Power
        forecast_probabilistic['lower_bound'] = q_power[parameter['len_compute']:,\
                                                        parameter['lb_quantiles'],hour]
        
        #Upper Bound for Power
        forecast_probabilistic['upper_bound'] = q_power[parameter['len_compute']:,\
                                                        parameter['ub_quantiles'],hour]
            
        #CDF
        for k1 in range(0, len(q_energy)-parameter['len_compute']):
            forecast_probabilistic['CDF'][k1] = estimate_CDF(q_energy[k1+parameter['len_compute'],:,day] - \
                                                             q_energy[k1+parameter['len_compute'],49,day], \
                                                             np.arange(0.01,1,0.01))
        
        assert((forecast_probabilistic['upper_bound'] - forecast_probabilistic['lower_bound']).all())
        return(forecast_median, forecast_probabilistic)
    else:
        return(forecast_median)
    
####################################Calculate CDF####################################      
#Define logistic functions
def log1_function(q, w):
    y = w[0]/(1+np.exp(-w[1]*(q-w[2])))
    return(y)

def log2_function(q, w):
    y = log1_function(q, w[0:3]) + log1_function(q, w[3:6])
    return(y)

#Estimate weights of logistic function
def estimate_weights(q, p, log_function):
    if log_function == log1_function:
        length_w = 3
    else:
        length_w = 6
    w = SX.sym('w', length_w, 1)
    objective = sum1((log_function(q,w) - p)**2)/len(p)
     
    nlp = {}
    nlp['x'] = w
    nlp['f'] = objective
    if log_function == log1_function:
        nlp['g'] = w[0]
        lower_g = [1]
        upper_g = [1]
        x0 = [1, 1, 1]
    else:
        nlp['g'] = vertcat(*[w[0], w[1], w[0] + w[3]])
        lower_g = [0,0,1]
        upper_g = [1,1,1]
        x0 = [0.5, 1, 1, 0.5, 1, 1]
    
    opts = {'ipopt.print_level':0, 'print_time':0}
    F = nlpsol('F','ipopt',nlp,opts)
    weights = F(x0 = x0, lbg = lower_g, ubg = upper_g)
    status = F.stats()['return_status']
   
    if status == 'Solve_Succeeded':
        result = {'weights': weights['x'], 'objective': weights['f']}
        return(result)
    else:
        return('Solve_not_Succeeded')
    
#Decide if single logistic function or sum of two logistic functions 
def decide_log_function(q, p):
    result_log2 = estimate_weights(q, p, log2_function)
    result_log1 = estimate_weights(q, p, log1_function)

    if type(result_log1) != str: 
        if type(result_log2) == str or result_log2['objective'] >= result_log1['objective'] or \
        -10 > result_log2['weights'][1] or result_log2['weights'][1] > 10 or -10 > result_log2['weights'][4] or \
        result_log2['weights'][4] > 10 or \
        sqrt(np.sum(log2_function(q, result_log2['weights']) - p)**2/len(q)) > 0.1:
            y = result_log1['weights']
        else:
            y = result_log2['weights']
    else:
        if type(result_log2) == str:
            y = 'No solution found'
        else:
            y = result_log2['weights']
    assert(y != str)
    return(y)

#Final function for estimating CDF
def estimate_CDF(q, p):
    weights = np.array(decide_log_function(q, p))
    if len(weights) == 3:
        def CDF(x):
            y = log1_function(x, weights)
            return(y)
    else:
        def CDF(x):
            y = log2_function(x, weights)
            return(y)
    return(CDF)


####################################Setting Parameter####################################  
parameter = {
    'costs':{'single_plus': 0.3, 'quadratic_plus': 0.05, 'single_minus': 0.15, 'quadratic_minus': 0.05, \
             'alpha': 50, 'gamma': [0.05] + [0.1] + [0.2] + [0.4] + [1000000]*400},
    'e_start': 6.75, 
    #delta is distance between steps -> 1 hour
    'delta': 1,
    'mu': 0.05,
    'p_min': -5,
    'p_max': 5,
    'e_min': 0,
    'e_max': 13.5,
    'epsilon': 0.2,
    #relaxation for p in 1stage_ComputeDispatchSchedule und 2stage_MPC
    'relaxation_p': 1e-8
}
parameter['epsilon_quantiles'] = 0.4
parameter['lb_quantiles'] = round(50 - (1-parameter['epsilon_quantiles'])/2*100 - 1)
parameter['ub_quantiles'] = round(50 + (1-parameter['epsilon_quantiles'])/2*100 - 1)
parameter['len_DS'] = 24
#DS can be extended up to 36 hours
parameter['len_DS_extended'] = 30
parameter['len_DS_extension'] = parameter['len_DS_extended'] - parameter['len_DS']
parameter['len_compute'] = 12
parameter['len_cycle'] = parameter['len_compute'] + parameter['len_DS_extended']
parameter['len_MPC'] = 6
#Number houses
parameter['number_houses'] = 1
