class Warehouse:    
    def __init__(self, fixedCost) -> None:
        self.fixed_cost = fixedCost
        self.is_open = False
        
    def __lt__(self, other):
        if not isinstance(other, Warehouse):
            return NotImplemented
        return self.fixed_cost < other.fixed_cost
        
    def __str__(self) -> str:
        return f'(Fixed Cost: {self.fixed_cost}, Open: {self.is_open})'
    
    def __repr__(self) -> str:
        return self.__str__()