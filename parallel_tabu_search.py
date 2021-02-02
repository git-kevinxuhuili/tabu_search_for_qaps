import time
import numpy as np 
import pandas as pd
import itertools as itr
import functools
from multiprocessing import Pool, Process
import multiprocessing

Letter = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def get_a_shuffled_letter_list(n):
    """Initialise a shuffled x0 vector

    :param n: size of letters (facilities)
    :return: a shuffled x0 vector
    """
    np.random.seed(n)
    multiplier = n // 100
    remainder = n % 100
    if n <= 100:
        letter = [Letter[0] + str(j) for j in range(remainder)]
    else:
        letter_1 = [Letter[i] + str(j) for i in range(multiplier) for j in range(100)]
        letter_2 = [Letter[multiplier] + str(i) for i in range(remainder)]
        letter = letter_1 + letter_2
    np.random.shuffle(letter)
    return letter

    
def get_Dist(n):
    """Create a customised distance matrix

    :param n: size of letters (facilities)
    :return: a customised distance matrix
    """
    rows = list()
    base = [i for i in range(0, int(n/2))]
    right = [(i+1) for i in base]
    rows.append(base + right)
    for j in range(1, int(n/2)):
        new_base = [abs(i-j) for i in base]
        new_right = [(i+1) for i in new_base]
        new_row = new_base + new_right
        rows.append(new_row)
    for k in range(int(n/2)-1, -1, -1):
        rows.append(rows[k][-1::-1])
    letter = get_a_shuffled_letter_list(n)
    df = pd.DataFrame(rows, columns = letter, index = letter)
    return df
    

def get_Flow(n):
    """Create a randomised flow matrix

    :param n: size of letters (facilities)
    :return: a randomised flow matrix with flows betweeen each facility less and equal to 10.
    """
    letter = get_a_shuffled_letter_list(n)
    np.random.seed(n)
    df = pd.DataFrame(np.random.randint(11, size=(n,n)), columns = letter, index = letter)
    for i in range(n):
        df.iloc[i,i] = 0
    return df


class Par_Tabu_Search:
    """
    Parallel tabu search
    """
    def __init__(self, X0 = None, Dist = None, Flow = None, Length_of_Tabu_List = 10):

        self.X0 = X0
        self.Dist = Dist
        self.Flow = Flow
        self.Length_of_Tabu_List = Length_of_Tabu_List
        self.Tabu_List = np.empty((0, len(X0) + 1))
        self.Save_Solutions_Here = np.empty((0, len(X0) + 1))
        self.Current_Sol = None
        self.List_of_N = None
        self.One_Final_Guy_Final= list()
        self.Final_Here = list()
        self.iteration_checker = 0


    def get_cost(self, X0):
        """Get overall cost for the quadratic assignment problem.
        
        Cost = Distance * Flow 

        :param: X0 vector 
        :return: overall cost
        """
        Flow_DF = self.Flow.reindex(columns=X0, index=X0) # Reindex the flow matrix 
        Flow_Arr = np.array(Flow_DF)

        Objfun1_start = pd.DataFrame(Flow_Arr*self.Dist) # Compute the overall cost
        Objfun1_start_Arr = np.array(Objfun1_start)

        sum_start_int = sum(sum(Objfun1_start_Arr))

        return sum_start_int


    def pre_search(self):
        """ Preliminary search phase (Short-term memory)
        """

        self.List_of_N = list(itr.combinations(self.X0, 2))  # From X0, it shows how many combinations of 2's can it get; 8 choose 2
        All_N_for_i = np.empty((0, len(self.X0)))
        List_of_N_numerical_list = [i for i in range(len(self.List_of_N))]
        
        # First multiprocessing
        X_Swap_List = p1.map(self.swap, List_of_N_numerical_list)
        All_N_for_i = np.vstack(X_Swap_List)

        """ Non-parellel version
        for j in range(len(List_of_N)):
            
            X_Swap = []
            A_Counter = List_of_N[j] # Goes through each set 
            A_1 = A_Counter[0]
            A_2 = A_Counter[1]

            # Making a new list of the new set of departments, with 2-opt (swap)
                        
            for l in range(len(X0)): # For elements in X0, swap the set chosen and store it
                
                if X0[l] == A_1:
                    X_Swap = np.append(X_Swap, A_2)
                elif X0[l] == A_2:
                    X_Swap = np.append(X_Swap, A_1)
                else:
                    X_Swap = np.append(X_Swap, X0[l])
                
                X_Swap = X_Swap[np.newaxis] # New "X0" after swap
                        
             All_N_for_i = np.vstack((All_N_for_i,X_Swap)) # Stack all the combinations
        """
        OF_Values_for_N_i = np.empty((0, len(self.X0) + 1)) # +1 to add the OF values
        OF_Values_all_N = np.empty((0, len(self.X0) + 1))   # +1 to add the OF values 

        # Calculate OF for the i-th solution in N
    
        # Second multiprocessing
        Total_Cost_N_i_lst = p2.map(self.get_cost, All_N_for_i)

        for k in range(len(All_N_for_i)):

            OF_Values_for_N_i = np.column_stack((Total_Cost_N_i_lst[k],All_N_for_i[k][np.newaxis])) # Stack the OF value to the department vector
            OF_Values_all_N = np.vstack((OF_Values_all_N,OF_Values_for_N_i))
    
        """ Non-parellel version
        for k in All_N_for_i:
            Total_Cost_N_i = get_cost(k, Dist = Dist, Flow = Flow)
            
            k = k[np.newaxis]
        
            OF_Values_for_N_i = np.column_stack((Total_Cost_N_i,k)) # Stack the OF value to the department vector
            OF_Values_all_N = np.vstack((OF_Values_all_N,OF_Values_for_N_i))
        """

        # Ordered OF of neighborhood, sorted by OF value
        OF_Values_all_N_Ordered = np.array(sorted(OF_Values_all_N, key=lambda x: x[0]))
        
        # Check if solution is already in Tabu list, if yes, choose the next one
        t = 0
        self.Current_Sol = OF_Values_all_N_Ordered[t] # Current solution   
        while self.Current_Sol[0] in self.Tabu_List[:,0]: # If current solution is in Tabu list
            self.Current_Sol = OF_Values_all_N_Ordered[t]
            t = t+1
    
        if len(self.Tabu_List) >= self.Length_of_Tabu_List: # If Tabu list is full
            self.Tabu_List = np.delete(self.Tabu_List, np.s_[(self.Length_of_Tabu_List-1):], axis=0) # Delete the last row
        
        self.Tabu_List = np.vstack((self.Current_Sol,self.Tabu_List)) # Save the best solution to the tabu list        
        self.Save_Solutions_Here = np.vstack((self.Current_Sol,self.Save_Solutions_Here)) # Save solutions, which is the best in each run
        self.X0 = self.Current_Sol[1:]


    def swap(self, x):

        X_Swap = []
        A_Counter = self.List_of_N[x] # Goes through each set 
        A_1 = A_Counter[0]
        A_2 = A_Counter[1]

        # Making a new list of the new set of departments, with 2-opt (swap)
        for l in range(len(self.X0)): # For elements in X0, swap the set chosen and store it
            if self.X0[l] == A_1:
                X_Swap = np.append(X_Swap, A_2)
            elif self.X0[l] == A_2:
                X_Swap = np.append(X_Swap, A_1)
            else:
                X_Swap = np.append(X_Swap, self.X0[l])
                    
                X_Swap = X_Swap[np.newaxis] # New "X0" after swap

        return X_Swap


    def exploitation(self):
        """ Exploitation phase (Intermediate-term memory)
        """ 
        min_cost = min(self.Tabu_List[:,0])
        if self.Current_Sol[0] > min_cost: 
            self.iteration_checker += 1
            if self.iteration_checker == self.Length_of_Tabu_List:
                self.X0 = np.array(sorted(self.Tabu_List, key=lambda x: x[0]))[0, 1:]
                self.iteration_checker = 0
        else: 
            self.iteration_checker = 0


    def exploration(self):
        """ Exploration phase (Long-term memory)
        """          
        # In order to "kick-start" the search when stuck in a local optimum, for diversification
        Ran = np.random.randint(3,len(self.X0))
        Ran_1 = Ran // 3 
        Ran_2 = 2 * Ran // 3 
        Ran_3 = 3 * Ran // 3 

        Xt = list()
        # Making a new list of the new set of departments
        S_Temp = self.Current_Sol
        for w in range(len(S_Temp)):
            if S_Temp[w] == self.Current_Sol[Ran_1]:
                Xt=np.append(Xt,self.Current_Sol[Ran_2])
            elif S_Temp[w] == self.Current_Sol[Ran_2]:
                Xt=np.append(Xt,self.Current_Sol[Ran_3])
            elif S_Temp[w] == self.Current_Sol[Ran_3]:
                Xt=np.append(Xt,self.Current_Sol[Ran_1])
            else:
                Xt=np.append(Xt,S_Temp[w])

        self.Current_Sol = Xt
        self.X0 = self.Current_Sol[1:]


    def modify_tabu_list(self):

        # Reordered Tabu List from the best to worst 
        self.Tabu_List = np.array(sorted(self.Tabu_List, key=lambda x: x[0]))
            
        # Change the pre-defined length of Tabu List every 25 runs, between 5 and 20, dynamic Tabu list
        np.random.seed(len(self.X0))
        self.Length_of_Tabu_List = np.random.randint(5,20)


    def find_optimal_solution(self):
        
        # Find the optimal answer from all iterations
        for i in range(len(self.Save_Solutions_Here)):
            if (self.Save_Solutions_Here[i,0]) <= min(self.Save_Solutions_Here[:,0]):
                Final_Here = self.Save_Solutions_Here[i,:]
        One_Final_Guy_Final = Final_Here[np.newaxis]
        
        print('\n')
        print("Min in all Iterations:",One_Final_Guy_Final[0])
        print("The Lowest Cost is:", One_Final_Guy_Final[0,0])


class Tabu_Search:
    """
    Non-parallel tabu search
    """
    def __init__(self, X0 = None, Dist = None, Flow = None, Length_of_Tabu_List = 10):

        self.X0 = X0
        self.Dist = Dist
        self.Flow = Flow
        self.Length_of_Tabu_List = Length_of_Tabu_List
        self.Tabu_List = np.empty((0, len(X0) + 1))
        self.Save_Solutions_Here = np.empty((0, len(X0) + 1))
        self.Current_Sol = None
        self.List_of_N = None
        self.One_Final_Guy_Final= [] 
        self.Final_Here = []
        self.iteration_checker = 0


    def get_cost(self, X0):
        """Get overall cost for the quardratic assignment problem.
        
        Cost = Distance * Flow 

        :param: X0 vector 
        :return: overall cost
        """
        Flow_DF = self.Flow.reindex(columns=X0, index=X0) # Reindex the flow matrix 
        Flow_Arr = np.array(Flow_DF)
        
        Objfun1_start = pd.DataFrame(Flow_Arr*self.Dist) # Compute the overall cost
        Objfun1_start_Arr = np.array(Objfun1_start)

        sum_start_int = sum(sum(Objfun1_start_Arr))

        return sum_start_int


    def pre_search(self):
        """ Preliminary search phase (Short-term memory)
        """
        self.List_of_N = list(itr.combinations(self.X0, 2))  # From X0, it shows how many combinations of 2's can it get; 8 choose 2
        
        All_N_for_i = np.empty((0, len(self.X0)))
        
        for j in range(len(self.List_of_N)):
            
            X_Swap = []
            A_Counter = self.List_of_N[j] # Goes through each set 
            A_1 = A_Counter[0]
            A_2 = A_Counter[1]

            # Making a new list of the new set of departments, with 2-opt (swap)
                        
            for l in range(len(self.X0)): # For elements in X0, swap the set chosen and store it
                
                if self.X0[l] == A_1:
                    X_Swap = np.append(X_Swap, A_2)
                elif self.X0[l] == A_2:
                    X_Swap = np.append(X_Swap, A_1)
                else:
                    X_Swap = np.append(X_Swap, self.X0[l])
                
                X_Swap = X_Swap[np.newaxis] # New "X0" after swap
                        
            All_N_for_i = np.vstack((All_N_for_i,X_Swap)) # Stack all the combinations  
        
        OF_Values_for_N_i = np.empty((0, len(self.X0) + 1)) # +1 to add the OF values
        OF_Values_all_N = np.empty((0, len(self.X0) + 1))   # +1 to add the OF values 
    
        # Calculate OF for the i-th solution in N        
        for k in All_N_for_i:
            Total_Cost_N_i = self.get_cost(k)
            
            k = k[np.newaxis]
        
            OF_Values_for_N_i = np.column_stack((Total_Cost_N_i,k)) # Stack the OF value to the department vector
            OF_Values_all_N = np.vstack((OF_Values_all_N,OF_Values_for_N_i))
        
        # Ordered OF of neighborhood, sorted by OF value
        OF_Values_all_N_Ordered = np.array(sorted(OF_Values_all_N, key=lambda x: x[0]))
        
        # Check if solution is already in Tabu list, if yes, choose the next one
        t = 0
        self.Current_Sol = OF_Values_all_N_Ordered[t] # Current solution 
        while self.Current_Sol[0] in self.Tabu_List[:,0]: # If current solution is in Tabu list
            self.Current_Sol = OF_Values_all_N_Ordered[t]
            t = t+1
    
        if len(self.Tabu_List) >= self.Length_of_Tabu_List: # If Tabu list is full
            self.Tabu_List = np.delete(self.Tabu_List, np.s_[(self.Length_of_Tabu_List-1):], axis=0) # Delete the last row
        
        self.Tabu_List = np.vstack((self.Current_Sol,self.Tabu_List))
        
        self.Save_Solutions_Here = np.vstack((self.Current_Sol,self.Save_Solutions_Here)) # Save solutions, which is the best in each run

        self.X0 = self.Current_Sol[1:]


    def swap(self, x):

        X_Swap = []
        A_Counter = self.List_of_N[x] # Goes through each set 
        A_1 = A_Counter[0]
        A_2 = A_Counter[1]

        # Making a new list of the new set of departments, with 2-opt (swap)
        for l in range(len(self.X0)): # For elements in X0, swap the set chosen and store it
            if self.X0[l] == A_1:
                X_Swap = np.append(X_Swap, A_2)
            elif self.X0[l] == A_2:
                X_Swap = np.append(X_Swap, A_1)
            else:
                X_Swap = np.append(X_Swap, self.X0[l])
                    
                X_Swap = X_Swap[np.newaxis] # New "X0" after swap

        return X_Swap


    def exploitation(self):
        """ Exploitation phase (Intermediate-term memory)
        """  
        min_cost = min(self.Tabu_List[:,0])
        #while i < len(self.Save_Solutions_Here):
        if self.Current_Sol[0] > min_cost: 
            self.iteration_checker += 1
            if self.iteration_checker == self.Length_of_Tabu_List:
                self.X0 = np.array(sorted(self.Tabu_List, key=lambda x: x[0]))[0, 1:]
                # self.Tabu_List = np.empty((0, len(self.X0) + 1))
                self.iteration_checker = 0
        else: 
            self.iteration_checker = 0


    def exploration(self):
        """ Exploration phase (Long-term memory)
        """        
        # In order to "kick-start" the search when stuck in a local optimum, for diversification
        Ran = np.random.randint(3,len(self.X0))
        Ran_1 = Ran // 3 
        Ran_2 = 2 * Ran // 3 
        Ran_3 = 3 * Ran // 3 

        Xt = list()
        # Making a new list of the new set of departments
        S_Temp = self.Current_Sol
        for w in range(len(S_Temp)):
            if S_Temp[w] == self.Current_Sol[Ran_1]:
                Xt=np.append(Xt,self.Current_Sol[Ran_2])
            elif S_Temp[w] == self.Current_Sol[Ran_2]:
                Xt=np.append(Xt,self.Current_Sol[Ran_3])
            elif S_Temp[w] == self.Current_Sol[Ran_3]:
                Xt=np.append(Xt,self.Current_Sol[Ran_1])
            else:
                Xt=np.append(Xt,S_Temp[w])

        self.Current_Sol = Xt
        self.X0 = self.Current_Sol[1:]


    def modify_tabu_list(self):

        # Reordered Tabu List from the best to worst 
        self.Tabu_List = np.array(sorted(self.Tabu_List, key=lambda x: x[0]))
            
        # Change the pre-defined length of Tabu List every 25 runs, between 5 and 20, dynamic Tabu list
        np.random.seed(len(self.X0))
        self.Length_of_Tabu_List = np.random.randint(5,20)


    def find_optimal_solution(self):
        
        # Find the optimal solution from all iterations
        for i in range(len(self.Save_Solutions_Here)):
            if (self.Save_Solutions_Here[i,0]) <= min(self.Save_Solutions_Here[:,0]):
                Final_Here = self.Save_Solutions_Here[i,:]
        One_Final_Guy_Final = Final_Here[np.newaxis]
        
        print('\n')
        print("Min in all Iterations:",One_Final_Guy_Final[0])
        print("The Lowest Cost is:", One_Final_Guy_Final[0,0])


if __name__ == '__main__':

    p1 = Pool(processes = 8)
    p2 = Pool(processes = 8)

    def play(Runs, Par = None, tabu = None): 
        '''
        parameters:

        Runs: how many iterations 

        Par:
        True: Parallel Tabu Search 
        False: Non-parallel Tabu Search

        tabu:
        Tabu_Search(...): Parallel tabu search algo
        Par_Tabu_Search(...): Non-parallel tabu search algo
        '''
        t0 = time.time()
        
        # Parallel Tabu Search 
        if Par == True:
            print('\n') 
            print('Parellel Tabu Search')
            for iteration in range(1, Runs+1):

                # print('--> This is the %i' % iteration, 'th iteration <--')

                ##############################################################################
                # Preliminary search phase
                tabu.pre_search()

                ##############################################################################
                # Exploitation phase 
                tabu.exploitation()

                if iteration % 25 == 0:
                    
                    # Third multiprocessing
                    ##############################################################################
                    # Exploration phase   
                    p3 = Process(target = tabu.exploration)
                    p3.start()
                    ##############################################################################
                    # Modify the order of tabu list and change length of Tabu List every 25 runs, between 5 and 20, dynamic Tabu list
                    p4 = Process(target = tabu.modify_tabu_list)
                    p4.start()

                    
            # Print Optimal Solution for all iterations 
            tabu.find_optimal_solution()

        # Non-parallel Tabu Search
        else: 
            print('\n') 
            print('Non-parellel Tabu Search')
            for iteration in range(1, Runs+1):

                # print('--> This is the %i' % iteration, 'th iteration <--')

                ##############################################################################
                # Preliminary search phase
                tabu.pre_search()

                ##############################################################################
                # Exploitation phase 
                tabu.exploitation()
        
                if iteration % 25 == 0:

                    ##############################################################################
                    # Exploration phase                      
                    tabu.exploration()
                    # Modify the order of tabu list and change length of Tabu List every 25 runs, between 5 and 20, dynamic Tabu list
                    tabu.modify_tabu_list()


            # Print Optimal Solution for all iterations 
            tabu.find_optimal_solution()

        t1 = time.time()
        total_time = t1 - t0
        print('\n')
        print('Total time it takes: ', round(total_time/60,3), 'minutes.')
        print('\n')

        return total_time

    
    def get_comparison_result(n):
        
        print('---------------------------')
        print('---------------------------')
        print('Size of n: ', n)
        print('---------------------------')
        print('---------------------------')
        
        ##############################################################################
        # Initialisation phase
        Dist = get_Dist(n) # Customised distance matrix 
        Flow = get_Flow(n) # Randomised flow matrix with the value of flow between 1 to 10
        X0 = get_a_shuffled_letter_list(n) # Initial x0 solution 
        Length_of_Tabu_List = 10 # Pre-defined length of tabu list (Like exploration phase, it changes for 25 iterations.)
        
        ##############################################################################
        # 500 iterations 
        total_time_1 = play(500, Par = False, tabu = Tabu_Search(X0 = X0, Dist = Dist, Flow = Flow, Length_of_Tabu_List = Length_of_Tabu_List)) # Time for non-parallel tabu search
        total_time_2 = play(500, Par = True,  tabu = Par_Tabu_Search(X0 = X0, Dist = Dist, Flow = Flow, Length_of_Tabu_List = Length_of_Tabu_List)) # Time for parallel tabu-search

        return [total_time_1, total_time_2]

    for i in range(8,52,2): # n is in (8, 10, 12, ..., 48, 50)
        get_comparison_result(i)
