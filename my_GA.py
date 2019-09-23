#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import random
import math
from operator import itemgetter
import matplotlib.pyplot as plt



class GA:
    '''
    This is a class of GA algorithm.
    '''

    def __init__(self, parameter):
        '''
        Initialize the pop of GA algorithom and evaluate the pop by computing its' fitness value .
        The data structure of pop is composed of several individuals which has the form like that:

        {'Gene':a object of class Gene, 'fitness': 1.02(for example)}
        Representation of Gene is a list: [b s0 u0 sita0 s1 u1 sita1 s2 u2 sita2]

        '''
        # parameter = [CXPB, MUTPB, NGEN, popsize, low, up]
        self.parameter = parameter

        low = self.parameter[4]
        up = self.parameter[5]

        self.bound = []
        self.bound.append(low)
        self.bound.append(up)

        pop = []
        for i in range(self.parameter[3]):
            geneinfo = []
            for pos in range(len(low)):
                geneinfo.append(random.uniform(self.bound[0][pos], self.bound[1][pos]))  # initialise popluation

               #fitness = evaluate(geneinfo)  # evaluate each chromosome
            fitness = 21.5+geneinfo[0]*math.sin(4.0*math.pi*geneinfo[0])+geneinfo[1]*math.sin(20.0*math.pi*geneinfo[1])  # evaluate each chromosome
            pop.append({'Gene': geneinfo, 'fitness': fitness})  # store the chromosome and its fitness

        self.pop = pop
        self.bestindividual = self.selectBest(self.pop)  # store the best chromosome in the population

    def selectBest(self, pop):
        '''
        select the best individual from pop
        '''
        s_inds = sorted(pop, key=itemgetter("fitness"), reverse=True)
        return s_inds[0]

    def selection(self, individuals, k):
        '''
        select two individuals from pop
	roulette_selected_population(zhaopeng)
        '''
        s_inds = sorted(individuals, key=itemgetter("fitness"),
                        reverse=True)  # sort the pop by the reference of 1/fitness
        sum_fits = sum(1 / ind['fitness'] for ind in individuals)  # sum up the 1/fitness of the whole pop

        fitness = []
        for ind in s_inds:
            fitness.append(ind['fitness'])
        fitness_array = np.array(fitness)
        print('fitness_array = ',fitness_array)
        fitness_distribution = np.exp(fitness_array - fitness_array[0])/np.sum(np.exp(fitness_array-fitness_array[0]))
        print("fitness = ",fitness_distribution)
        pop_size = len(individuals)
        pop_indexes = np.random.choice(pop_size,k,p=fitness_distribution)
        #print("individuals = ",pop_indexes)

        chosen = []
        for i in pop_indexes:
            chosen.append(s_inds[i])
        #print("chosen = ",chosen)
        return chosen

    def crossoperate(self, pop ,elite=0):
        '''
        cross operation
        @param elite defines how many chromosomes remain unchanged
        '''
        pop_size = len(pop)
        if(elite > pop_size): 
            ValueError("Error: the elite value cannot be larger than the population size")
        new_pop = []
        # cope the elite into the new pop
        for i in range(0,elite):
            new_pop.append(pop[i])

        #print("elite = ",elite)
        # random pick the present to cross
        presents_index = np.random.randint(low=0, high=pop_size,size=(pop_size-elite,2))
        #print("presents_index = ",presents_index)

        # Generating the remaining individuals through crossover
        for ind in range(0,pop_size-elite):
            first_present = pop[presents_index[ind,0]]
            second_present = pop[presents_index[ind,1]]
            dim = len(first_present)
            
            pos1 = random.randrange(0,dim,1)  ## select a position in the range from 0 to dim-1,
            pos2 = random.randrange(0,dim,1)  ## select a position in the range from 0 to dim-1,
            newoff = []
            tmp = []
            for i in range(dim):
                if(i>min(pos1,pos2)) and (1<max(pos1,pos2)):
                    tmp.append(second_present[i])
                else:
                    tmp.append(first_present[i])
                newoff = tmp
            #print('first = ',first_present)
            #print('second = ',second_present)
            #print('newoff = ',newoff)
            new_pop.append(newoff)
        #print('pop = ',pop)
        #print('newpop = ',new_pop)
        return new_pop

    def mutation(self, crossoff_Gene, bound, elite=0):
        '''
        mutation operation
        '''
        dim = len(crossoff_Gene[0])
        pop_size = len(crossoff_Gene)
        MUTPB = self.parameter[1]
        #print('crossoff_Gen[0] = ',crossoff_Gene)

        pos = random.randrange(0, dim, 1)  # chose a position in crossoff to perform mutation.
        for i in range(elite,pop_size):
            if(np.random.uniform(0,1)<MUTPB):
                #print('before crossoff_Gen[i] = ',crossoff_Gene[i])
                crossoff_Gene[i][pos] = random.uniform(bound[0][pos], bound[1][pos])
                #print('after crossoff_Gen[i] = ',crossoff_Gene[i])

        #print('crossoff_Gen[0] = ',crossoff_Gene)
        return crossoff_Gene


    def GA_main(self):
        '''
        evaluation frame work of GA
        '''

        popsize = self.parameter[3]
        elite = self.parameter[6]
        mean_fitness_list = []

        print("Start of evolution")

        # Begin the evolution
        for g in range(NGEN):

            print("-- Generation %i --" % g)

            # Apply selection based on their converted fitness
            selectpop = self.selection(self.pop, popsize)

            selectpop_Gene = [ind['Gene'] for ind in selectpop]

            crossoff = self.crossoperate(selectpop_Gene,elite)

            mutation = self.mutation(crossoff, self.bound, elite)
            # Gather all the fitnesses in one list and print the stats
            next_pop = []
            for ind in range(0,popsize):
                mutation_s = mutation[ind]
                fitness = 21.5+mutation_s[0]*math.sin(4.0*math.pi*mutation_s[0])+mutation_s[1]*math.sin(20.0*math.pi*mutation_s[1])  # evaluate each chromosome
                next_pop.append({'Gene':mutation[ind],'fitness':fitness})

            # The population is entirely replaced by the offspring
            self.pop = next_pop
            fits = [ind['fitness'] for ind in self.pop]
            #print('self.pop = ',self.pop)


            length = len(self.pop)
            mean = sum(fits) / length
            mean_fitness_list.append(mean)
            sum2 = sum(x * x for x in fits)
            std = abs(sum2 / length - mean ** 2) ** 0.5
            best_ind = self.selectBest(self.pop)

            if best_ind['fitness'] > self.bestindividual['fitness']:
                self.bestindividual = best_ind

            print(
                "Best individual found is %s, %s" % (self.bestindividual['Gene'], self.bestindividual['fitness']))
            print("  Min fitness of current pop: %s" % min(fits))
            print("  Max fitness of current pop: %s" % max(fits))
            print("  Avg fitness of current pop: %s" % mean)
            print("  Std of currrent pop: %s" % std)
        array = np.arange(1,NGEN+1,dtype='int32')
        plt.plot(array, mean_fitness_list,  color='red', marker='o', markersize=6, markevery=10, label='Mean')
        plt.xlim(0, 80)
        plt.xlabel('Generation', fontsize=15)
        #plt.yticks(np.linspace(34,39,0.5,endpoint=True))
        plt.ylabel('Fitness', fontsize=15)
        print("Saving the image in './fitness.jpg'...")
        plt.savefig("./fitness.jpg", dpi=500)

        print("-- End of (successful) evolution --")


if __name__ == "__main__":
    CXPB, MUTPB, NGEN, popsize,elite = 0.5, 0.1, 100, 200, 2  # control parameters

    up = [12.1,5.8]  # upper range for variables
    low = [-3.0,4.1]  # lower range for variables
    parameter = [CXPB, MUTPB, NGEN, popsize, low, up, elite]

    run = GA(parameter)
    run.GA_main()
