import json

class Params:
    def __init__(self, params_dict: dict = {}):
        for key, val in params_dict.items():
            self.__dict__[key] = val
    
    def __str__(self):
        return json.dumps(self.__dict__, indent = 4)
    
    def __setattr__(self, key: str, val: any):
        self.__dict__[key] = val
    
    def __getattr__(self, key: str):
        if key in self.__dict__:
            return self.__dict__[key]
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")
    
    def save(self, file_path: str = "diags/params.json"):
        with open(file_path, "w") as fp:
            json.dump(self.__dict__, fp)
    
    def load(self, file_path: str = "diags/params.json"):
        with open(file_path, "r") as fp:
            params_dict = json.load(fp)
            for key, val in params_dict.items():
                self.__dict__[key] = val
            
if __name__ == "__main__":
    import os
    print("testing")
    
    print("test1: assignment")
    p = Params({"a": 1})
    p.b = 2
    assert p.a == 1
    assert p.b == 2
    print("test1 pass")
    
    print("test2: save & load")
    p.save("test-params.json")
    p.load("test-params.json")
    assert p.a == 1
    assert p.b == 2
    print("test2 pass")
    os.remove("test-params.json")
    