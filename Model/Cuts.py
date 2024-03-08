class Cuts:
    def __init__(self, id, cost, A, Ac ):
        self.id = id
        self.cost = cost
        self.A = A
        self.Ac = Ac

    def is_in(self, v):
        return v in self.A

    
    
   