import random

class GeneticAlgorithmNoSearch:

    def __init__(self, warehouses, customers, population_size=500, generations=100, crossover_rate=0.8, mutation_rate=0.1, tournament_size=5, elitism=True, seed=None, diversity_threshold=0.1, diversity_mutation_increase=0.5, random_immigrants_rate=0.1):
        """
        Initializes a GeneticAlgorithmWOLocalSearch object.

        Parameters:
        - warehouses (list): A list of warehouses.
        - customers (list): A list of customers.
        - population_size (int): The size of the population (default: 50).
        - generations (int): The number of generations to run the algorithm (default: 100).
        - crossover_rate (float): The probability of crossover (default: 0.8).
        - mutation_rate (float): The probability of mutation (default: 0.1).
        - tournament_size (int): The size of the tournament selection (default: 5).
        - elitism (bool): Whether to use elitism in the selection process (default: True).
        - seed (int): The seed for the random number generator (default: None).
        - diversity_threshold (float): The threshold for diversity preservation (default: 0.1).
        - diversity_mutation_increase (float): The increase in mutation rate for diversity preservation (default: 0.5).
        - random_immigrants_rate (float): The rate of random immigrants in each generation (default: 0.1).
        """
        self.warehouses = warehouses
        self.customers = customers
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.diversity_threshold = diversity_threshold
        self.diversity_mutation_increase = diversity_mutation_increase
        self.random_immigrants_rate = random_immigrants_rate
        if seed is not None:
            random.seed(seed)
        self.population = self.initialize_population()

   
    def initialize_population(self):
        """
        Initializes the population for the genetic algorithm.

        Returns:
            list: A list of binary lists representing the population.
                  Each binary list represents a solution, where True
                  indicates that a warehouse is selected and False
                  indicates that a warehouse is not selected.
        """
        return [[random.choice([True,False]) for _ in range(len(self.warehouses))] for _ in range(self.population_size)]

   
    def calculate_cost(self, solution):
        """
        Calculates the total cost of a given solution.

        Parameters:
        - solution (list): A list representing the solution, where each element indicates whether a facility is open or not.

        Returns:
        - total_cost (float): The total cost of the solution.
        """
        total_cost = 0
        for i, facility_open in enumerate(solution):
            if facility_open:
                total_cost += float(self.warehouses[i].fixed_cost)
        for customer in self.customers:
            min_cost = float('inf')
            for i, facility_open in enumerate(solution):
                if facility_open:
                    min_cost = min(min_cost, float(customer.costs[i]))
            if min_cost == float('inf'):
                return float('inf')
            total_cost += min_cost
        return total_cost

    
    def evaluate_population(self):
        """
        Evaluates the fitness of each individual in the population.

        Returns:
            A list of fitness values for each individual in the population.
        """
        return [self.calculate_cost(individual) for individual in self.population]

   
    def tournament_selection(self, fitness):
        """
        Performs tournament selection to select individuals from the population based on their fitness.

        Args:
            fitness (list): A list of fitness values corresponding to each individual in the population.

        Returns:
            list: A list of selected individuals.

        """
        selected = []
        for _ in range(self.population_size):
            tournament = random.sample(range(self.population_size), self.tournament_size)
            tournament_fitness = [(self.population[i], fitness[i]) for i in tournament]
            tournament_fitness.sort(key=lambda x: x[1])
            selected.append(tournament_fitness[0][0])
        return selected

  
    def crossover(self, parent1, parent2):
        """
        Performs crossover between two parent solutions.

        Args:
            parent1 (list): The first parent solution.
            parent2 (list): The second parent solution.

        Returns:
            tuple: A tuple containing two child solutions resulting from the crossover operation.
        """
        if random.random() < self.crossover_rate:
            point = random.randint(1, len(self.warehouses) - 1)
            return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
        return parent1, parent2

 
    def mutate(self, individual):
        """
        Mutates an individual by flipping the values of its elements randomly based on the mutation rate.

        Parameters:
            individual (list): The individual to be mutated.

        Returns:
            list: The mutated individual.
        """
        new_individual = individual[:]
        for i in range(len(new_individual)):
            if random.random() < self.mutation_rate:
                new_individual[i] = not new_individual[i]
        return new_individual

 
    def calculate_diversity(self):
        """
        Calculates the diversity of the population.

        Returns:
            float: The diversity of the population as a ratio of unique individuals to the population size.
        """
        unique_individuals = {tuple(individual) for individual in self.population}
        return len(unique_individuals) / self.population_size

  
    def introduce_random_immigrants(self):
        """
        Introduces random immigrants into the population.

        Random immigrants are individuals with random binary values representing the presence or absence of warehouses.
        The number of immigrants is determined by the population size and the random immigrants rate.

        Returns:
            None
        """
        num_immigrants = int(self.population_size * self.random_immigrants_rate)
        for _ in range(num_immigrants):
            self.population[random.randint(0, self.population_size - 1)] = [random.choice([True, False]) for _ in range(len(self.warehouses))]

    
    def run(self):
        """
        Runs the genetic algorithm with or without local search.

        Returns:
            Tuple: A tuple containing the best solution and its cost.
        """
        best_solution = None
        best_cost = float('inf')

        for _ in range(self.generations):
            fitness = self.evaluate_population()
            current_best_cost = min(fitness)
            current_best_solution = self.population[fitness.index(current_best_cost)]

            if current_best_cost < best_cost:
                best_cost = current_best_cost
                best_solution = current_best_solution
            

            parents = self.tournament_selection(fitness)
            next_population = []

            if self.elitism:
                next_population.append(best_solution)  

            while len(next_population) < self.population_size:
                parent1, parent2 = random.sample(parents, 2)
                offspring1, offspring2 = self.crossover(parent1, parent2)
                next_population.append(self.mutate(offspring1))
                if len(next_population) < self.population_size:
                    next_population.append(self.mutate(offspring2))

            self.population = next_population

            
            diversity = self.calculate_diversity()
            if diversity < self.diversity_threshold:
                
                self.mutation_rate += self.diversity_mutation_increase
            else:
                self.mutation_rate = max(self.mutation_rate - self.diversity_mutation_increase, 0.01)

            self.introduce_random_immigrants()

        return best_solution, best_cost
