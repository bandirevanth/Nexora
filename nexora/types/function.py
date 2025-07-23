import json

class Function:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = json.dumps(arguments)

    def __repr__(self):
        return "Function(name='{}', parameters={})".format(self.name, repr(self.arguments))
