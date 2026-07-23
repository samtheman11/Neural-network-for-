import numpy as np

class Data :
    def __init__(self,Baseline=1000):
        self.Baseline=Baseline    
      
    def daily_pattern(self,hour):# 0 = midnight, 23 = 11pm
        """
        This function given an hour of the day from 0 to 23 
        returns a demand multiplier which is based on the time of day.
        since the power demand is lower around night time and higher during the day,
        this can be used to simulate the daily pattern of power demand.
        """
        if hour < 0 or hour > 23: #Stops invalid input
            raise ValueError("Hour must be between 0 and 23")
        clock_A = np.cos(2 * np.pi * (hour - 15) / 24) 
        clock_B = np.cos(2 * np.pi * (hour - 8) / 12) 
        shift = 1.1
        scaler = 2.5
        combined = clock_A + 0.5 * clock_B
        return (combined/scaler) + shift # Normalize to a range of 0 to 1

    def DayOfWeekMultiplier(self,day):#0=Monday, 1=Tuesday, 2=Wednesday, 3=Thursday, 4=Friday, 5=Saturday, 6=Sunday
        """
        Since power demand is lower on the weekends than on weekdays
        due to less industrial and commerical usage, this returns a multiplier based off real data of the day of the week.
        """
        if day < 0 or day > 6: #Stops invalid input
            raise ValueError("Day must be between 0 and 6")
        else:
            if day >= 5: #the weekend
                return 0.85
            else: return 1 # weekdays

    def sim_temperature(self,day_of_year):
        """
        This function simulates the temperature based on the day of the year.
        The coldest day is in January(0) and the Hottest day is in July(180)
        """
        if day_of_year < 0 or day_of_year > 364: #Stops invalid input
            raise ValueError("Day of year must be between 0 and 364")
        else:
            scaler = 35
            shift = 55
            noise = np.random.normal(0, 2) # Add some random noise to the temperature
            return np.cos(2 * np.pi * (day_of_year - 180)/365)*scaler + shift + noise

    def DayOfYearMultiplier(self,temp):#day of year 0-364 
        """
        This functions return a multiplier for the day of the year.
        The coldest day is in January(0) and the Hottest day is in July(180)
        """
        if temp > 75:#cooling
            return (temp - 75)*np.random.uniform(.02,.04)+1#used ai to find percent increase per temp when cooling  
            
        elif temp < 65:#heating
            return (65-temp)*np.random.uniform(.03,.05)+1#used ai to find percent increase per temp when heating
        else:
            return 1          
    def dataset(self,num_hours):
        hour_array = []
        day_array = []
        demand_array = []
        temp_array=[]
        for i in range(num_hours):
            hour = i % 24
            day = (i//24)%7
            temp = self.sim_temperature((i//24)%365)
            
            hour_array.append(hour)
            day_array.append(day)
            temp_array.append(temp)

            noise = np.random.normal(0,10)
            d = self.Baseline*self.daily_pattern(hour)*self.DayOfWeekMultiplier(day)*self.DayOfYearMultiplier(temp)
            demand_array.append(d+noise)

        X = np.column_stack([hour_array,day_array,temp_array])
        Y = np.array(demand_array)
        return X,Y





