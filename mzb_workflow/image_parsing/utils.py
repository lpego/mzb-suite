
class cfg_to_arguments(object):
    # This class is used to convert a dictionary to an object and extend the argparser      
    def __init__(self, args):
        for key in args:
            setattr(self, key, args[key])
                    
    def __str__(self):
        return self.__dict__.__str__()
    
