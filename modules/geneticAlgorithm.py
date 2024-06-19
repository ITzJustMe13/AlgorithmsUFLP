import random
from concurrent.futures import ProcessPoolExecutor
from collections import deque

class GeneticAlgorithm:
    
    def __init__(self, warehouses, customers, population_size=50, generations=100, crossover_rate=0.8, mutation_rate=0.1, tournament_size=5, elitism=True, seed=None, diversity_threshold=0.1, diversity_mutation_increase=0.5, random_immigrants_rate=0.1):
        """
        Initializes a GeneticAlgorithm object.

        Parameters:
        - warehouses (list): A list of warehouses.
        - customers (list): A list of customers.
        - population_size (int): The size of the population (default: 50).
        - generations (int): The number of generations to run the algorithm (default: 100).
        - crossover_rate (float): The probability of crossover occurring during reproduction (default: 0.8).
        - mutation_rate (float): The probability of mutation occurring during reproduction (default: 0.1).
        - tournament_size (int): The size of the tournament selection (default: 5).
        - elitism (bool): Whether to use elitism in the selection process (default: True).
        - seed (int): The seed value for the random number generator (default: None).
        - diversity_threshold (float): The threshold for measuring population diversity (default: 0.1).
        - diversity_mutation_increase (float): The increase in mutation rate when population diversity is low (default: 0.5).
        - random_immigrants_rate (float): The rate of random immigrants introduced in each generation (default: 0.1).
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
            population (list): A list of individuals representing the initial population.
        """
        population = []
        for _ in range(self.population_size):
            while True:
                individual = [random.choice([True,False]) for _ in range(len(self.warehouses))]
                if any(individual): 
                    population.append(individual)
                    break
        return population

    
    def calculate_cost(self, solution):
        """
        Calculates the total cost of a given solution.

        Parameters:
        - solution (list): A binary list representing the solution, where each element indicates whether a facility is open or closed.

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
        Evaluates the fitness of each individual in the population using parallel processing.

        Returns:
            fitness (list): A list of fitness values for each individual in the population.
        """
        with ProcessPoolExecutor() as executor:
            fitness = list(executor.map(self.calculate_cost, self.population))
        return fitness

    def tournament_selection(self, fitness):
        """
        Selects individuals from the population using tournament selection.

        Args:
            fitness (list): A list of fitness values for each individual in the population.

        Returns:
            list: A list of selected individuals.

        """
        selected = []
        for _ in range(self.population_size):
            tournament = random.sample(range(self.population_size), self.tournament_size)
            tournament_fitness = [fitness[i] for i in tournament]
            best = tournament[tournament_fitness.index(min(tournament_fitness))]
            selected.append(self.population[best])
        return selected

    
    def crossover(self, parent1, parent2):
        """
        Performs crossover between two parent individuals.

        Args:
            parent1 (list): The first parent individual.
            parent2 (list): The second parent individual.

        Returns:
            tuple: A tuple containing two offspring individuals resulting from the crossover.

        """
        if random.random() < self.crossover_rate:
            point = random.randint(1, len(parent1) - 2)
            return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
        else:
            return parent1, parent2
 
    
    def mutate(self, solution):
        """
        Mutates the given solution by flipping the values of some genes based on the mutation rate.

        Args:
            solution (list): The solution to be mutated, represented as a list of boolean values.

        Returns:
            list: The mutated solution.

        """
        mutated = [not gene if random.random() < self.mutation_rate else gene for gene in solution]
        if not any(mutated):  
            mutated[random.randint(0, len(mutated) - 1)] = True
        return mutated

    def tabu_search(self, solution, max_iterations=100, tabu_tenure=10):
        """
        Perform tabu search algorithm to find the best solution for the given problem.

        Args:
            solution (list): The initial solution to start the search from.
            max_iterations (int): The maximum number of iterations to perform.
            tabu_tenure (int): The maximum number of solutions to keep in the tabu list.

        Returns:
            list: The best solution found by the tabu search algorithm.
        """
        best_solution = solution[:]
        best_cost = self.calculate_cost(solution)
        current_solution = solution[:]
        current_cost = best_cost

        tabu_list = deque(maxlen=tabu_tenure)

        for _ in range(max_iterations):
            neighbors = self.generate_neighbors(current_solution)
            best_neighbor = None
            best_neighbor_cost = float('inf')

            for neighbor in neighbors:
                neighbor_tuple = tuple(neighbor)
                if neighbor_tuple in tabu_list:
                    continue
                neighbor_cost = self.calculate_cost(neighbor)
                if neighbor_cost < best_neighbor_cost:
                    best_neighbor = neighbor
                    best_neighbor_cost = neighbor_cost

            if best_neighbor is None:
                break

            current_solution = best_neighbor
            current_cost = best_neighbor_cost
            tabu_list.append(tuple(current_solution))

            if current_cost < best_cost:
                best_solution = current_solution
                best_cost = current_cost

        return best_solution

   
    def generate_neighbors(self, solution):
        """
        Generates neighboring solutions by flipping the state of each facility in the given solution.

        Args:
            solution (list): The current solution represented as a list of binary values.

        Returns:
            list: A list of neighboring solutions, where each solution is obtained by flipping the state of one facility in the given solution.

        """
        neighbors = []
        for i in range(len(solution)):
            neighbor = solution[:]
            neighbor[i] = not neighbor[i]
            if any(neighbor):  
                neighbors.append(neighbor)
        return neighbors

    
    def calculate_diversity(self):
        """
        Calculates the diversity of the population.

        The diversity is calculated as the ratio of unique individuals to the total population size.

        Returns:
            float: The diversity of the population.
        """
        unique_individuals = {tuple(individual) for individual in self.population}
        return len(unique_individuals) / self.population_size


    def introduce_random_immigrants(self):
        """
        Introduces random immigrants into the population.

        This method randomly selects a number of individuals from the population and replaces their genes with random values.

        Parameters:
        - None

        Returns:
        - None
        """
        num_immigrants = int(self.population_size * self.random_immigrants_rate)
        for _ in range(num_immigrants):
            self.population[random.randint(0, self.population_size - 1)] = [random.choice([True, False]) for _ in range(len(self.warehouses))]

 
    def run(self):
        """
        Runs the genetic algorithm to find the best solution.

        Returns:
            tuple: A tuple containing the best solution and its cost.
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
                next_population.append(self.tabu_search(self.mutate(offspring1)))
                if len(next_population) < self.population_size:
                    next_population.append(self.tabu_search(self.mutate(offspring2)))

            self.population = next_population

            diversity = self.calculate_diversity()
            if diversity < self.diversity_threshold:
                self.mutation_rate += self.diversity_mutation_increase
            else:
                self.mutation_rate = max(self.mutation_rate - self.diversity_mutation_increase, 0.01)

            self.introduce_random_immigrants()

        return best_solution, best_cost
