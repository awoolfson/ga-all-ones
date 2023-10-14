import numpy as np
import random
from pprint import pprint
import matplotlib.pyplot as plt

CHROMOSOME_SIZE = 64
POPULATION_SIZE = 100
MUTATION_RATE = 0.001
GENERATIONS = 2000
X = 10 # save max and average fitness every X generations

SAVE_INITIAL_POP = True
READ_INITIAL_POP = False
SAVE_FINAL_POP = True

def main():
    population = np.ndarray(shape=(POPULATION_SIZE, CHROMOSOME_SIZE), dtype=int)
    if READ_INITIAL_POP:
        with open("inputpop.txt", "r") as f:
            for i in range(POPULATION_SIZE):
                try:
                    population[i] = np.fromiter(f.readline().strip(), dtype=int)
                except:
                    print("error reading inputpop.txt")
                    exit()
            f.close()
    else:
        for i in range(POPULATION_SIZE):
            population[i] = np.random.randint(2, size=CHROMOSOME_SIZE)
        
    pprint(population)

    if SAVE_INITIAL_POP:
        with open("inputpop.txt", "w") as f:
            for i in range(POPULATION_SIZE):
                f.write("".join(str(x) for x in population[i]) + "\n")
            f.close()

    max_fitnesses = np.ndarray(shape=(GENERATIONS // X,), dtype=int)
    avg_fitnesses = np.ndarray(shape=(GENERATIONS // X,), dtype=int)
    most_fits = np.ndarray(shape=(GENERATIONS // X, CHROMOSOME_SIZE), dtype=int)
    saving_idx = 0
        
    # run GA
    for generation in range(GENERATIONS):
        
        print(f"generation {generation}")
        
        # calculate fitness
        raw_fitness = np.ndarray(shape=(POPULATION_SIZE,), dtype=int)
        for i, chromosome in enumerate(population):
            raw_fitness[i] = sum(chromosome)
            
        fitness = np.ndarray(shape=(POPULATION_SIZE,), dtype=int)
        for i, raw in enumerate(raw_fitness):
            fitness[i] = raw ** 2
            
        most_fit = population[max(range(len(fitness)), key=fitness.__getitem__)].copy()
        mutation_bound = int(1 / MUTATION_RATE)
        
        # fitnesses are actually the square of the amount of 1s, but it is easier to make sense of the results this way
        if generation % X == 0:
            max_fitness = sum(most_fit)
            avg_fitness = sum(raw_fitness) / len(raw_fitness)
            max_fitnesses[saving_idx] = max_fitness
            avg_fitnesses[saving_idx] = avg_fitness
            most_fits[saving_idx] = most_fit
            print(f"max fitness: {max_fitness}")
            print(f"average fitness: {avg_fitness}")
            saving_idx += 1
        
        # select parents
        roulette_wheel = np.ndarray(shape=(POPULATION_SIZE,), dtype=int)
        last = 0
        for i in range(len(fitness)):
            roulette_wheel[i] = fitness[i] + last
            last = roulette_wheel[i]
        
        parents_idx = np.ndarray(shape=(POPULATION_SIZE,), dtype=int)
        
        # binary search roulette wheel for parents
        for i in range(POPULATION_SIZE):
            spin = random.randrange(0, roulette_wheel[-1])
            l, r = 0, len(roulette_wheel) - 1
            while l <= r:
                m = l + (r - l) // 2
                if roulette_wheel[m] == spin:
                    parents_idx[i] = m
                    break
                elif roulette_wheel[m] > spin:
                    if m > 0 and roulette_wheel[m - 1] < spin:
                        parents_idx[i] = m
                        break
                    r = m - 1
                else:
                    l = m + 1
            parents_idx[i] = l
            
        # generate offspring population
        for i in range(len(parents_idx) // 2):
            
            mom = population[parents_idx[i * 2]]
            dad = population[parents_idx[i * 2 + 1]]
            partition = random.randrange(0, CHROMOSOME_SIZE)
            
            for j in range(CHROMOSOME_SIZE):
                mutation = random.randrange(0, mutation_bound)
                mutate_first = True if mutation == 0 else False
                mutation = random.randrange(0, mutation_bound)
                mutate_second = True if mutation == 0 else False
                
                if j <= partition:
                    population[i * 2][j] = mom[j] if not mutate_first else 1 - mom[j]
                    population[i * 2 + 1][j] = dad[j] if not mutate_second else 1 - dad[j]
                else:
                    population[i * 2][j] = dad[j] if not mutate_first else 1 - dad[j]
                    population[i * 2 + 1][j] = mom[j] if not mutate_second else 1 - mom[j]
                    
        population[0] = most_fit
                    
    plt.plot(range(GENERATIONS // X), max_fitnesses)
    plt.title("Max Fitness by Generation")
    plt.xlabel(f"Generation (every {X})")
    plt.ylabel("Max Fitness")
    plt.show()

    plt.plot(range(GENERATIONS // X), avg_fitnesses)
    plt.title("Average Fitness by Generation")
    plt.ylabel("Average Fitness")
    plt.show()

    if SAVE_FINAL_POP:
        with open("outputpop.txt", "w") as f:
            for i in range(POPULATION_SIZE):
                f.write("".join(str(x) for x in population[i]) + "\n")
            f.close()
            
if __name__ == "__main__":
    main()
