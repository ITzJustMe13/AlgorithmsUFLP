class Warehouse:    
    def __init__(self, fixedCost) -> None:
        """
        Initializes a Warehouse object.

        Args:
            fixedCost (float): The fixed cost associated with the warehouse.

        Attributes:
            fixed_cost (float): The fixed cost associated with the warehouse.
            is_open (bool): Indicates whether the warehouse is open or closed.
        """
        self.fixed_cost = fixedCost
        self.is_open = False
        
    def __lt__(self, other):
        """
        Compare two Warehouse objects based on their fixed_cost attribute.

        Args:
            other (Warehouse): The other Warehouse object to compare with.

        Returns:
            bool: True if self.fixed_cost is less than other.fixed_cost, False otherwise.
        """
        if not isinstance(other, Warehouse):
            return NotImplemented
        return self.fixed_cost < other.fixed_cost
        
    def __str__(self) -> str:
        """
        Returns a string representation of the Warehouse object.

        The string includes the fixed cost and the open status of the warehouse.

        Returns:
            str: A string representation of the Warehouse object.
        """
        return f'(Fixed Cost: {self.fixed_cost}, Open: {self.is_open})'
    
    def __repr__(self) -> str:
        """
        Returns a string representation of the object.
            
        This method is used to provide a string representation of the object
        that can be used for debugging and logging purposes. It should return
        a string that uniquely identifies the object and provides enough
        information to recreate the object if possible.
            
        Returns:
            str: A string representation of the object.
        """
        return self.__str__()