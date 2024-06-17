import random
from multiprocessing import Pool

class GeneticAlgorithm:
    def __init__(self, warehouses, customers, population_size=50, generations=100, crossover_rate=0.8, mutation_rate=0.1, tournament_size=5, elitism=True, seed=None, diversity_threshold=0.1, diversity_mutation_increase=0.5, random_immigrants_rate=0.1):
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
        self.fixed_costs = [float(warehouse.fixed_cost) for warehouse in self.warehouses]
        self.customer_costs = [customer.costs for customer in self.customers]
        self.cost_matrix = self.compute_cost_matrix()

    def initialize_population(self):
        return [[random.choice([True,False]) for _ in range(len(self.warehouses))] for _ in range(self.population_size)]

    def compute_cost_matrix(self):
        cost_matrix = []
        for solution in self.population:
            cost_matrix.append(self.calculate_cost(solution))
        return cost_matrix

    def calculate_cost(self, solution):
        total_cost = 0
        for i, facility_open in enumerate(solution):
            if facility_open:
                total_cost += self.fixed_costs[i]
        for customer_cost in self.customer_costs:
            min_cost = min([cost for open_, cost in zip(solution, customer_cost) if open_])
            total_cost += float(min_cost)
        return total_cost

    def evaluate_population(self):
        with Pool() as pool:
            return pool.map(self.calculate_cost, self.population)

    def tournament_selection(self, fitness):
        selected = []
        for _ in range(self.population_size):
            tournament = random.sample(range(self.population_size), self.tournament_size)
            tournament_fitness = [(self.population[i], fitness[i]) for i in tournament]
            tournament_fitness.sort(key=lambda x: x[1])
            selected.append(tournament_fitness[0][0])
        return selected

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            point = random.randint(1, len(self.warehouses) - 1)
            return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
        return parent1, parent2

    def mutate(self, individual):
        new_individual = individual[:]
        for i in range(len(new_individual)):
            if random.random() < self.mutation_rate:
                new_individual[i] = not new_individual[i]
        return new_individual

    def calculate_diversity(self):
        unique_individuals = {tuple(individual) for individual in self.population}
        return len(unique_individuals) / self.population_size

    def introduce_random_immigrants(self):
        num_immigrants = int(self.population_size * self.random_immigrants_rate)
        for _ in range(num_immigrants):
            self.population[random.randint(0, self.population_size - 1)] = [random.choice([True, False]) for _ in range(len(self.warehouses))]

    def local_search(self, individual):
        best_solution = individual[:]
        best_cost = self.calculate_cost(individual)

        while True:
            neighbors = self.generate_neighbors(best_solution)
            found_better = False

            for neighbor in neighbors:
                neighbor_cost = self.calculate_cost(neighbor)
                if neighbor_cost < best_cost:
                    best_solution = neighbor
                    best_cost = neighbor_cost
                    found_better = True

            if not found_better:
                break

        return best_solution

    def generate_neighbors(self, solution):
        neighbors = []

        for i in range(len(solution)):
            neighbor = solution[:]
            neighbor[i] = not neighbor[i]
            neighbors.append(neighbor)

        return neighbors
    
    def run(self):
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
                next_population.append(self.local_search(self.mutate(offspring1)))
                if len(next_population) < self.population_size:
                    next_population.append(self.local_search(self.mutate(offspring2)))

            self.population = next_population

            diversity = self.calculate_diversity()
            if diversity < self.diversity_threshold:
                self.mutation_rate += self.diversity_mutation_increase
            else:
                self.mutation_rate = max(self.mutation_rate - self.diversity_mutation_increase, 0.01)

            self.introduce_random_immigrants()

        return best_solution, best_cost
