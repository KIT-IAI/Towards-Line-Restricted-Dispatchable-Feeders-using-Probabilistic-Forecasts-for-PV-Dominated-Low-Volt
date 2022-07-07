from utils import house

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
    
####################################Run All Deterministic####################################
def run_all_deterministic_variablehouses_function(parameter, houses, quantiles, true_power):
    
    ##Initialization
    forecast_median = dict.fromkeys(['house' + str(i) for i in range(len(houses))], None)
    actual_power = dict.fromkeys(['house' + str(i) for i in range(len(houses))], None)
    estimate_e = dict.fromkeys(['house' + str(i) for i in range(len(houses))], None)
    actual_e = dict.fromkeys(['house' + str(i) for i in range(len(houses))], None)
    DS = dict.fromkeys(['house' + str(i) for i in range(len(houses))], None)
    stage1_costs = []
    stage2_costs = []
    stage3_costs = []
    
    ##########################Scheduling##########################

    ####Day 0
    ###hour = 12:00

    ##Generate Forecasts for Median for intervall 12:00-13:00 upwards
    for i, house in enumerate(houses):
        forecast_median['house' + str(i)] = calculate_forecast_quantities(quantiles['house' + str(i)]['power'], \
                                                                          quantiles['house' + str(i)]['energy'], 0)

    ##Estimate SoE in 00:00 -> not needed in Day 0
    #Save results
    for i, house in enumerate(houses):
        actual_e['house' + str(i)] = house.e_start
        house.estimate_e['value'][0] = house.e_start
        house.actual_e['value'][0] = house.e_start 

    ##1. Stage: Compute DS; DS from 00:00 - 23:00 + extension 
    forecast_median_extract = {keys: values[parameter['len_compute']:] for keys, values in forecast_median.items()}
    stage1_costs.append(compute_DS(parameter, houses, forecast_median_extract, actual_e))
    #Save results
    for i, house in enumerate(houses):
        house.dec_var_to_results('stage1', 0, parameter['len_DS'])
        house.stage1_forecast_power['value'][0:parameter['len_DS']] = \
        forecast_median['house' + str(i)][parameter['len_compute']:parameter['len_compute']+parameter['len_DS']]


    ####Day = 1,... 
    ###hour = 00:00,... 
    #index_time: index of actual time
    #time: actual time
    for index_time, time in enumerate(houses[0].hours[int(np.where(houses[0].hours == houses[0].start_schedule)[0]):], start=0):

        #index_forecast: index of actual time in forecast data
        index_forecast = index_time + parameter['len_compute']

        ##Generate Forecasts for Median from time upwards
        for i in range(len(houses)):
            forecast_median['house' + str(i)] = calculate_forecast_quantities(quantiles['house' + str(i)]['power'], \
                                                                              quantiles['house' + str(i)]['energy'], \
                                                                              index_forecast)


        ##########################Offline##########################

        ##If 12:00: Estimate SoE and Compute DS
        if time.hour == 12 and time.day != houses[0].end_compute.day:

            #index_schedule: Start of schedule
            index_schedule = int(np.where((houses[0].index_DS.hour == 0) & \
                                          (houses[0].index_DS.day == time.day + 1))[0])
            index_day = np.where(time.day == houses[0].days)[0]

            ##Estimate SoE in 00:00
            #We need forecasts from 12:00 upwards
            #We need DS from 12:00 upwards
            #e is SoE in 12:00
            for i, house in enumerate(houses):
                DS['house' + str(i)] = house.stage1_g['value'][index_time:]
            estimate_e = estimate_SOE(parameter, houses, forecast_median, actual_e, DS)
            #Save results
            for i, house in enumerate(houses):
                house.estimate_e['value'][index_day] = estimate_e['house' + str(i)]
       
            
            ##1. Stage: Compute DS from 00:00 - 23:00
            #We need forecasts from 00:00 upwards 
            #e0 is SoE in 00:00
            forecast_median_extract = {keys: values[parameter['len_compute']:] for keys, values in forecast_median.items()}
            stage1_costs.append(compute_DS(parameter, houses, forecast_median_extract, estimate_e))
            #Save results
            for i, house in enumerate(houses):
                house.dec_var_to_results('stage1', index_schedule, parameter['len_DS_extended'])
                house.stage1_forecast_power['value'][index_schedule:index_schedule+parameter['len_DS']] = \
                forecast_median['house' + str(i)][parameter['len_compute']:parameter['len_compute']+parameter['len_DS']]
                
        ##########################Online##########################
        
        
        ##2. Stage: MPC
        #We need forecasts from time upwards 
        #We need DS from time
        #e is SoE in time
        for i, house in enumerate(houses):
            house.stage2_forecast_power['value'][index_time:index_time+parameter['len_MPC']] = \
            forecast_median['house' + str(i)][0:parameter['len_MPC']]
        stage2_costs.append(mpc(parameter, houses, forecast_median, actual_e, index_time))
        #Save results
        for house in houses:
            house.dec_var_to_results('stage2', index_time, 1)
        
        ##Get actual load power
        for i, house in enumerate(houses):
            actual_power['house' + str(i)] = [true_power['house' + str(i)][index_forecast]]
            DS['house' + str(i)] = house.stage2_g['value'][index_time]

        ##3. Stage: Controlled ESS
        stage3_costs.append(controlled_ESS(parameter, houses, actual_power, actual_e, DS))
        #Save results
        for i, house in enumerate(houses):
            house.dec_var_to_results('actual', index_time, 1)
            house.actual_power['value'][index_time] = true_power['house' + str(i)][index_forecast]
            actual_e['house' + str(i)] = house.actual_e['value'][index_time+1]
        
    return(stage1_costs, stage2_costs, stage3_costs)


####################################Run All Perfect####################################
def run_all_perfect_variablehouses_function(parameter, houses, true_power):
    
    ##Initialization
    actual_e = dict.fromkeys(['house' + str(i) for i in range(len(houses))], None)
    stage1_costs = []
    
    ##########################Scheduling##########################

    ####Day 0
    ###hour = 12:00

    ##Estimate SoC in 00:00 -> not needed in Day 0
    #Save results
    for i, house in enumerate(houses):
        actual_e['house' + str(i)] = house.e_start

    ##1. Stage: Compute DS; DS from 00:00 - 23:00 + extension 
    true_power_extract = {keys: values[parameter['len_compute']:] for keys, values in true_power.items()}
    stage1_costs.append(compute_DS(parameter, houses, true_power_extract, actual_e))
    #Save results
    for i, house in enumerate(houses):
        house.dec_var_to_results('actual', 0, parameter['len_DS_extended'])


    ####Day = 1,... 
    ###hour = 00:00,... 
    #index_time: index of actual time
    #time: actual time
    for index_time, time in enumerate(houses[0].hours[int(np.where(houses[0].hours == houses[0].start_schedule)[0]):], start=0):

        #index_forecast: index of actual time in forecast data
        index_forecast = index_time + parameter['len_compute']

        ##########################Offline##########################

        ##If 12:00: Estimate SoC and Compute DS
        if time.hour == 12 and time.day != houses[0].end_compute.day:

            #index_schedule: Start of schedule
            index_schedule = int(np.where((houses[0].index_DS.hour == 0) & \
                                          (houses[0].index_DS.day == time.day + 1))[0])

            ##SoC in 00:00
            for i, house in enumerate(houses):
                actual_e['house' + str(i)] = house.actual_e['value'][index_schedule]
    
            
            ##1. Stage: Compute DS from 00:00 - 23:00
            #We need forecasts from 00:00 upwards 
            #e0 is SoC in 00:00
            true_power_extract = {keys: values[index_forecast+parameter['len_compute']:] for keys, values in true_power.items()}
            stage1_costs.append(compute_DS(parameter, houses, true_power_extract, actual_e))
            #Save results
            for i, house in enumerate(houses):
                house.dec_var_to_results('actual', index_schedule, parameter['len_DS_extended'])
        
        ##Get actual load power
        for i, house in enumerate(houses):
            house.actual_power['value'][index_time] = true_power['house' + str(i)][index_forecast]

    return(stage1_costs, stage2_costs, stage3_costs)


####################################Compute Dispatch Schedule####################################
def compute_DS(parameter, houses, forecast_median_extract, estimate_e):
    #Define variables
    objective = 0
    #Initialize constraints
    constraints_list = []
    lb_list = []
    ub_list = []
    #Initialize list of all decision variables
    decision = []
    
    for house in houses:
        house.g = SX.sym('g', parameter['len_DS_extended'])
        house.g_plus = SX.sym('g+', parameter['len_DS_extended'])
        house.g_minus = SX.sym('g-', parameter['len_DS_extended'])
        house.p = SX.sym('p', parameter['len_DS_extended'])
        house.p_plus = SX.sym('p+', parameter['len_DS_extended'])
        house.p_minus = SX.sym('p-', parameter['len_DS_extended'])
        house.e = SX.sym('e', parameter['len_DS_extended']+1)
        #Extend list of all decision variables
        decision.extend(house.dec_var_to_list('stage1'))
    
        for k in range(0, parameter['len_DS_extended']):
            objective = objective + (parameter['costs']['single_plus']*house.g_plus[k] + \
                                     parameter['costs']['quadratic_plus']*house.g_plus[k]**2 + \
                                     parameter['costs']['single_minus']*house.g_minus[k] + \
                                     parameter['costs']['quadratic_minus']*house.g_minus[k]**2) 
                            
        
            #g(k) = g+(k) + g-(k)
            constraint_split_g = house.g[k] - house.g_plus[k] - house.g_minus[k]
            lb_split_g = 0
            ub_split_g = 0
            constraints_list.append(constraint_split_g)
            lb_list.append(lb_split_g)
            ub_list.append(ub_split_g)

            #g+(k) >= 0 
            constraint_g_plus = house.g_plus[k]
            lb_g_plus = 0
            ub_g_plus = parameter['g_max']
            constraints_list.append(constraint_g_plus)
            lb_list.append(lb_g_plus)
            ub_list.append(ub_g_plus)

            #g-(k) <= 0
            constraint_g_minus = house.g_minus[k]
            lb_g_minus = -parameter['g_max']
            ub_g_minus = 0
            constraints_list.append(constraint_g_minus)
            lb_list.append(lb_g_minus)
            ub_list.append(ub_g_minus)                 
        
    #Add remaining constraints via define-constraints()
    #(houses, constraints_list, lb_list, ub_list, length, forecast_median, e)
    define_constraints(houses, constraints_list, lb_list, ub_list, parameter['len_DS_extended'], forecast_median_extract,\
                       estimate_e)
    
    #Initialize optimization problem
    nlp = {}
    nlp['x'] = vertcat(*decision)
    nlp['f'] = objective
    nlp['g'] = np.asarray(constraints_list)
    lower_bound = np.asarray(lb_list)
    upper_bound = np.asarray(ub_list)
    
    #Solve optimization problem
    opts = {'ipopt.print_level':0, 'print_time':0, 'ipopt.max_iter':10000000, 'ipopt.tol': 1e-11}
    F = nlpsol('F','ipopt',nlp,opts)
    opti = F(x0 = np.zeros(vertcat(*decision).shape[0]), lbg = lower_bound, ubg = upper_bound)
    opti_decision_list = np.squeeze(opti['x'])
    
    #Unravel results
    it1 = iter(opti_decision_list) 
    sizes = [item.shape[0] for item in decision]
    it2 = iter([[next(it1) for _ in range(size)] for size in sizes])                 
    for house in houses:
        liste = [next(it2) for _ in range(len(house.dec_var('stage1')))]                
        house.list_to_dec_var('stage1', liste)
    
    assert F.stats()['return_status'] == 'Solve_Succeeded', \
    "Assertion in 1stage_ComputeDispatchSchedule:" + F.stats()['return_status']
    return(np.squeeze(opti['f']))