def decorator(func):  
    def wrapper(*args, **kwargs):  
        print("Function is called with arguments:", args, kwargs)  
        result = func(*args, **kwargs)  
        return result  
    return wrapper  
  
@decorator  
def my_function(x, y):  
    return x + y  
  
print(my_function(3, 4))