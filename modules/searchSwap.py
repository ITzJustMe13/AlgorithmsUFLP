import random
import multiprocessing

class LocalSearchSwap:

    def __init__(self, warehouses, customers):
        self.warehouses = warehouses
        self.customers = customers
        self.current_solution = [random.choice([True, False]) for _ in range(len(warehouses))]
        if not any(self.current_solution):
            self.current_solution[random.randint(0, len(warehouses) - 1)] = True
        self.best_solution = self.current_solution[:]
        self.best_cost = self.calculate_cost(self.current_solution)
    
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
    
    def generate_neighbors(self, max_neighbors=10):
        neighbors = []
        open_indices = [i for i, open_state in enumerate(self.current_solution) if open_state]
        closed_indices = [i for i, open_state in enumerate(self.current_solution) if not open_state]
        
        for _ in range(min(max_neighbors, len(open_indices) * len(closed_indices))):
            open_idx = random.choice(open_indices)
            closed_idx = random.choice(closed_indices)
            neighbor_solution = self.current_solution[:]
            neighbor_solution[open_idx], neighbor_solution[closed_idx] = (
                neighbor_solution[closed_idx], neighbor_solution[open_idx]
            )
            neighbors.append(neighbor_solution)
        
        return neighbors

    def evaluate_neighbor(self, neighbor):
        return self.calculate_cost(neighbor)
    
    def quick_evaluate(self, solution):
        return sum(solution)
    
    def is_valid_solution(self, solution):
        return any(solution)

    def local_search(self, max_iterations_without_improvement=10):
        improvement = True
        iterations_without_improvement = 0
        
        while improvement and iterations_without_improvement < max_iterations_without_improvement:
            improvement = False
            neighbors = self.generate_neighbors()
            
            
            quick_scores = [self.quick_evaluate(neighbor) for neighbor in neighbors]
            
            
            sorted_neighbors = sorted(zip(quick_scores, neighbors), key=lambda x: x[0])
            
           
            with multiprocessing.Pool() as pool:
                detailed_scores = pool.map(self.evaluate_neighbor, [neighbor for _, neighbor in sorted_neighbors[:10]])
            
            for i, neighbor_cost in enumerate(detailed_scores):
                if neighbor_cost < self.best_cost:
                    self.best_solution = sorted_neighbors[i][1]
                    self.best_cost = neighbor_cost
                    improvement = True
                    iterations_without_improvement = 0
                else:
                    iterations_without_improvement += 1
            
            self.current_solution = self.best_solution[:]
        
        return self.best_solution, self.best_cost