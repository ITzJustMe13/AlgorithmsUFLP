import random

class GeneticAlgorithmNoSearch:

    '''Inicialização da classe, population_size é o numero de individuos por geração DEFAULT = 100, 
    generations é o numero de iterações que o código vai ter DEFAULT=100 , 
    crossover_rate é a frequencia que vai haver filhos DEFAULT=0.8/ 80%,
    mutation_rate é a frequencia que vai haver mutação durante o crossover DEFAULT=0.1/ 10%,
    tournament_size é o tamanho do torneio de seleção de pais DEFAULT= 5,
    elitism é se existe elitismo no algoritmo DEFAULT= TRUE,
    diversity_threshold é o maximo de diversidade dentro do algoritmo DEFAULT = 0.1 10%
    diversity_mutation_increase é o aumento de diversidade nas mutações DEFAULT= 0.5 50%
    random_immigrants_rate é o quão frequente imigrantes são inseridos DEFAULT= 0.1 10%
    '''
    def __init__(self, warehouses, customers, population_size=500, generations=100, crossover_rate=0.8, mutation_rate=0.1, tournament_size=5, elitism=True, seed=None, diversity_threshold=0.1, diversity_mutation_increase=0.5, random_immigrants_rate=0.1):
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

    #Inicializa a população
    def initialize_population(self):
        return [[random.choice([True,False]) for _ in range(len(self.warehouses))] for _ in range(self.population_size)]

    #Calcula os custos
    def calculate_cost(self, solution):
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

    #Avalia a população
    def evaluate_population(self):
        return [self.calculate_cost(individual) for individual in self.population]

    #Seleção de pais por torneio
    def tournament_selection(self, fitness):
        selected = []
        for _ in range(self.population_size):
            tournament = random.sample(range(self.population_size), self.tournament_size)
            tournament_fitness = [(self.population[i], fitness[i]) for i in tournament]
            tournament_fitness.sort(key=lambda x: x[1])
            selected.append(tournament_fitness[0][0])
        return selected

    #Criação de filhos (Crossover)
    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            point = random.randint(1, len(self.warehouses) - 1)
            return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
        return parent1, parent2

    #Função de Mutação
    def mutate(self, individual):
        new_individual = individual[:]
        for i in range(len(new_individual)):
            if random.random() < self.mutation_rate:
                new_individual[i] = not new_individual[i]
        return new_individual

    #Calcular diversidade
    def calculate_diversity(self):
        unique_individuals = {tuple(individual) for individual in self.population}
        return len(unique_individuals) / self.population_size

    #Introduz imigrantes
    def introduce_random_immigrants(self):
        num_immigrants = int(self.population_size * self.random_immigrants_rate)
        for _ in range(num_immigrants):
            self.population[random.randint(0, self.population_size - 1)] = [random.choice([True, False]) for _ in range(len(self.warehouses))]

    #RUN
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
