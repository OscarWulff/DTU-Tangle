from Cut import Cut

class DataType:
    def __init__(self,  agreement_param, cuts : list[Cut] = None,search_tree=None):
        self.agreement_param = agreement_param
        self.cuts = cuts
        self.search_tree = search_tree # perhaps
       

