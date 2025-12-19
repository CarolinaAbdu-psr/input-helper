import psr.factory

study = psr.factory.load_study("D:\\01-Repositories\\factory-graphs\\Bolivia")


#-------------------------------------
# Find object by property value
#-------------------------------------

def check_lower(value_a, value_b):
    return value_a < value_b

def check_equal(value_a, value_b):
    return value_a == value_b

def check_greatter(value_a, value_b):
    return value_a > value_b


def find_by_property_condition(type, property_name, property_condition, condition_value):
    """retuns all objects that match the condition: lower(l), equal(e), greatter(g)"""
    objects = []
    all_objects = study.find(type)
    for obj in all_objects:
        value = obj.get(property_name) #verificar se é estático     
        match = False
        if property_condition=="l":
            match = value < condition_value
        elif property_condition=="e":
            match = value == condition_value
        elif property_condition=="g":
            match = value > condition_value

        if match:
            objects.append(obj)
    return objects


#-------------------------------------
# Sum by condition 
#-------------------------------------

def sum_by_property_condition(type, property_name,property_condition, condition_value):
    sum = 0
    all_objects = study.find(type)
    for obj in all_objects:
        value = obj.get(property_name) #verificar se é estático     
        match = False
        if property_condition=="l":
            match = check_lower(value,condition_value)
        elif property_condition=="e":
            match = check_equal(value,condition_value)
        elif property_condition=="g":
            match = check_greatter(value,condition_value)

        if match:
            sum+= value 
    return sum

#-------------------------------------
# Count by condition 
#-------------------------------------
def count_by_property_condition(type, property_name,property_condition, condition_value):
    count = 0
    all_objects = study.find(type)
    for obj in all_objects:
        value = obj.get(property_name) #verificar se é estático     
        match = False
        if property_condition=="l":
            match = check_lower(value,condition_value)
        elif property_condition=="e":
            match = check_equal(value,condition_value)
        elif property_condition=="g":
            match = check_greatter(value,condition_value)

        if match:
            count+=1
    return count



#-------------------------------------
# Find by reference 
#-------------------------------------

def check_refererence(refs, reference_name):
    match = False
    for ref in refs: 
        if ref.name.strip() == reference_name:
            return True
    return match


def find_by_reference(type, reference_type, reference_name):
    objects = []
    all_objects = study.find(type)
    for obj in all_objects:
        refs = obj.get(reference_type) #Ex: RefFuels  
        print(refs)
        if not isinstance(refs, list):
            refs = [refs]
        match = check_refererence(refs,reference_name)
        if match:
            objects.append(obj)
        
    return objects

#-------------------------------------
# Count by reference 
#-------------------------------------
def count_by_reference(type, reference_type, reference_name):
    count = 0 
    all_objects = study.find(type)
    for obj in all_objects:
        refs = obj.get(reference_type) #Ex: RefFuels  
        print(refs)
        if not isinstance(refs, list):
            refs = [refs]
        match = check_refererence(refs,reference_name)
        if match:
            count+= 1
    return count

#-------------------------------------
# Sum property by reference 
#------------------------------------

def sum_property_by_reference(type, reference_type, reference_name, property):
    sum = 0 
    all_objects = study.find(type)
    for obj in all_objects:
        refs = obj.get(reference_type) #Ex: RefFuels  
        if not isinstance(refs, list):
            refs = [refs]
        match = check_refererence(refs,reference_name)
        if match:
            sum += obj.get(property)
    return sum 