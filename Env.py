# Import routines

import numpy as np
import math
import random

# Defining hyperparameters
m = 5 # number of cities, ranges from 0 ..... m-1
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger

lamda_A = 2    #lambda for poisson distribution to sample number of requests
lamda_B = 12
lamda_C = 4
lamda_D = 7
lamda_E = 8


class CabDriver():
    
    def __init__(self, m=m, t=t, d=d, C=C, R=R):
        
        """
            Initialise your state and define your action space and state space
            
        """
        
        self.action_space = [(0,0),(0,1),(0,2),(0,3),(0,4),(1,0),(1,2),(1,3),(1,4),(2,0),(2,1),(2,3),
                             (2,4),(3,0),(3,1),(3,2),(3,4),(4,0),(4,1),(4,2),(4,3)]                           
        self.state_space =  [0,1,2,3,4]                             
        
        # Initialization of all defined hyperparamters
        
        init_state = np.random.choice(np.arange(0,4))
        init_day = np.random.choice(np.arange(0,d))
        init_hour = np.random.choice(np.arange(0,t))
        
        self.state_init = (init_state, init_hour, init_day)
        self.m = m
        self.t = t
        self.d = d
        self.C = C
        self.R = R

        # Start the first round
        self.reset()

        
    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""

        #one hot encoding location
        location=np.zeros(5)
        np.put(location,[state[0]],[1])

        #one hot encoding time
        time=np.zeros(24)
        np.put(time,[state[1]],[1])

        #one hot encoding day
        day=np.zeros(7)
        np.put(day,[state[2]],[1])
    
        #concatenate all the one-hot encoded arrays
        state_encod=np.concatenate((location,time,day),axis=None)
     
        return state_encod



    # Use this function if you are using architecture-2 
    # def state_encod_arch2(self, state, action):
    #     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""

        
    #     return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        # Get location from the state variable
        location = state[0]

        if location == 0:
         
            requests = np.random.poisson(lamda_A)
        elif location == 1:
           
            requests = np.random.poisson(lamda_B)
        elif location == 2:
           
            requests = np.random.poisson(lamda_C)
        elif location == 3:
       
            requests = np.random.poisson(lamda_D)
        elif location == 4:
           
            requests = np.random.poisson(lamda_E)

        # As per requirement, the requests have to be limited to a maximum of 15
        if requests > 15:
            requests = 15
       
        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) # (0,0) is not considered as customer request
   
        actions = [self.action_space[i] for i in possible_actions_index]

        
        actions.append((0,0))   # Add the (0,0) action to the list

        return possible_actions_index,actions   



    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        if (action[0] == 0 and action[1] == 0): # no-ride action
            reward = -self.C
        else: #pick up and drop locations (𝑝,𝑞) where p and q both take a value between 0 and m-1
            
            time_taken_to_reach_pickup_loc = int(Time_matrix[state[0]][action[0]][state[1]][state[2]])
           
            hour_after_arrival_at_pickup_loc = state[1] + time_taken_to_reach_pickup_loc
            
            if(hour_after_arrival_at_pickup_loc >= 24):
                hour_after_arrival_at_pickup_loc = hour_after_arrival_at_pickup_loc % 24
                day_after_arrival_at_pickup_loc = state[2] + 1
                
                if (day_after_arrival_at_pickup_loc > 6 ):   # if day = 7 then day is set to 0
                    day_after_arrival_at_pickup_loc = 0      # day is reset to 0 (monday)
            else:    
                day_after_arrival_at_pickup_loc = state[2]   # still the same day
                
            updated_state_after_reaching_pickup_loc = (action[0], hour_after_arrival_at_pickup_loc, day_after_arrival_at_pickup_loc)
            
            
            time_p_to_q = int(Time_matrix[action[0]][action[1]][hour_after_arrival_at_pickup_loc][day_after_arrival_at_pickup_loc])

            
            reward = self.R*(time_p_to_q) - self.C*((time_taken_to_reach_pickup_loc) + (time_p_to_q))
            
       
        return reward




    def next_state_func(self, state, action, Time_matrix, total_trip_hours):
        """Takes state and action as input and returns next state"""
 
        if (action[0] == 0 and action[1] == 0):   # no-ride action
            location = state[0]                   #location remains the same
            hour =  state[1] + 1
            if (hour >= 24):                      # if hour cross 24 ,it mean its next day
                hour = hour % 24
                day = state[2] + 1
                if (day > 6 ):                    # if day = 7 then day is set to 0
                    day = 0                       # day is reset to 0 (monday)
            else:    
                day = state[2]                    # after moving the time component by 1 hour , its still same day
            total_hours = total_trip_hours + 1
            
            next_state = (location,hour,day,total_hours)   
            
        else:                                    #pick up and drop locations (𝑝,𝑞) where p and q both take a value between 1 and m
            location = action[1]
            
            #Fetching the total time taken to reach from one point to other from the Time Matrix
            time_taken_to_reach_pickup_loc = int(Time_matrix[state[0]][action[0]][state[1]][state[2]])
            hour_after_arrival_at_pickup_loc = state[1] + time_taken_to_reach_pickup_loc
            if(hour_after_arrival_at_pickup_loc >= 24):
                hour_after_arrival_at_pickup_loc = hour_after_arrival_at_pickup_loc % 24
                day_after_arrival_at_pickup_loc = state[2] + 1
                if (day_after_arrival_at_pickup_loc > 6 ):   # if day = 7 then day is set to 0
                    day_after_arrival_at_pickup_loc = 0      # day is reset to 0 (monday)
            else:    
                day_after_arrival_at_pickup_loc = state[2]   # still the same day
                
            updated_state_after_reaching_pickup_loc = (action[0], hour_after_arrival_at_pickup_loc, day_after_arrival_at_pickup_loc)
                
            
            time_taken_to_reach_destination =  int(Time_matrix[action[0]][action[1]][hour_after_arrival_at_pickup_loc][day_after_arrival_at_pickup_loc])
            # sum the existing hour with the total_time_taken_to_reach_destination
            hour = hour_after_arrival_at_pickup_loc + time_taken_to_reach_destination
            if (hour >= 24): # if hour cross 24 ,it mean its next day
                hour = hour % 24
                day = day_after_arrival_at_pickup_loc + 1
                if (day > 6 ):                              # if day = 7 then day is set to 0
                    day = 0                                 # day is reset to 0 (monday)
            else:    
                day = day_after_arrival_at_pickup_loc       # after dropping to destination, its still same day
            

            
            total_hours = total_trip_hours + time_taken_to_reach_pickup_loc + time_taken_to_reach_destination
        
            next_state = (location,hour,day,total_hours)
        return next_state




    def reset(self):
        return self.action_space, self.state_space, self.state_init
