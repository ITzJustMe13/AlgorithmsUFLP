class LocalSearchSwitch:

    def __init__(self, warehouses, customers):
        self.warehouses = warehouses
        self.customers = customers
        self.current_solution = [i % 2 == 0 for i in range(len(warehouses))]
        self.best_solution = self.current_solution[:]
        self.best_cost = self.calculate_cost(self.current_solution)
    
    #Função calcular custos
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
            total_cost += min_cost
        return total_cost
    
    #Gera vizinhos
    def generate_neighbors(self):
        neighbors = []
        for i in range(len(self.warehouses)):
            neighbor_solution = self.current_solution[:]
            neighbor_solution[i] = not neighbor_solution[i]  
            neighbors.append(neighbor_solution)
        return neighbors

    #Pesquisa Local
    def local_search(self):
        improvement = True
        while improvement:
            improvement = False
            neighbors = self.generate_neighbors()
            for neighbor in neighbors:
                neighbor_cost = self.calculate_cost(neighbor)
                if neighbor_cost < self.best_cost:
                    self.best_solution = neighbor
                    self.best_cost = neighbor_cost
                    improvement = True
            self.current_solution = self.best_solution[:]
        return self.best_solution, self.best_cost