#week 1
#Overall computation from 18-03-2013 12:00 - 25-03-2013 23:00
start_compute = pd.to_datetime('2013-03-18 12:00')
end_compute = pd.to_datetime('2013-03-25 23:00')
quantile_files = 
true_power_files =
#week 2
#Overall computation from 16-03-2013 12:00 - 23-04-2013 23:00
start_compute = pd.to_datetime('2013-04-16 12:00')
end_compute = pd.to_datetime('2013-04-23 23:00')
quantile_files = 
true_power_files =
#week 3
#Overall computation from 21-05-2013 12:00 - 28-05-2013 23:00
start_compute = pd.to_datetime('2013-05-21 12:00')
end_compute = pd.to_datetime('2013-05-28 23:00')
quantile_files = 
true_power_files = 


#Initialize Houses
houses = []
for _ in range(parameter['number_houses']):
    houses.append(ProbabilisticHouse(parameter, start_compute, end_compute))
quantile_files = ['Quantiles_modified_3G_both.pkl']
true_power_files = ['TruePower_modified_3G.pkl']
quantiles = dict.fromkeys(['house' + str(i) for i in range(len(houses))], None)
true_power = dict.fromkeys(['house' + str(i) for i in range(len(houses))], None)
for i, house in enumerate(houses):
    
    quantile_file = open(quantile_files[i], 'rb')
    quantiles['house' + str(i)] = {}
    quantiles['house' + str(i)] = pickle.load(quantile_file)
    
    true_power['house' + str(i)] = {}
    true_power_file = open(true_power_files[i], 'rb')
    true_power['house' + str(i)] = pickle.load(true_power_file)
    
    #Set battery capacity
    house.p_min = -5
    house.p_max = 5
    house.e_min = 0
    house.e_max = 13.5
    house.e_start = 13.5/2
    
    
#Perform Simulation
houses_unrestricted = copy.deepcopy(houses)
houses_restricted_error = copy.deepcopy(houses)
parameter['g_max'] = 200000
costs1_unrestricted, costs2_unrestricted, costs3_unrestricted = \
run_all_probabilistic_variablehouses_function(parameter, houses_unrestricted, quantiles, true_power)
houses_restricted = copy.deepcopy(houses_unrestricted)
g_max_unrestricted = max(abs(sum(house.actual_g['value'] for house in houses_unrestricted)))
parameter['g_max'] = g_max_unrestricted
print(parameter['g_max'])
while True:
    try:
        houses_restricted_temp = copy.deepcopy(houses) 
        parameter['g_max'] = round(parameter['g_max'], 1) - 0.2
        costs1_restricted, costs2_restricted, costs3_restricted = \
        run_all_probabilistic_variablehouses_function(parameter, houses_restricted_temp, quantiles, true_power)
        houses_restricted = copy.deepcopy(houses_restricted_temp)
        print(parameter['g_max'])
    except AssertionError as error:
        print(error, "with g_max =", parameter['g_max'])
        parameter['g_max'] = parameter['g_max'] + 0.1
        houses_restricted_error = copy.deepcopy(houses_restricted_temp)
        try:
            houses_restricted_temp = copy.deepcopy(houses)
            costs1_restricted, costs2_restricted, costs3_restricted = \
            run_all_probabilistic_variablehouses_function(parameter, houses_restricted_temp, quantiles, true_power)
            houses_restricted = copy.deepcopy(houses_restricted_temp)
            print(parameter['g_max'])
        except AssertionError as error:
            print(error, "with g_max =", parameter['g_max'])
            houses_restricted_error = copy.deepcopy(houses_restricted_temp)
            break
        break