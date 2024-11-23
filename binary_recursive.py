


def recursive(n, list):
    if n <= 0:
        return list
    else:
       reminder =  n % 2
       actual = n//2
       list.append(reminder)
       recursive (actual, list)
    return list


list = []
list = recursive(348, list)
list.reverse()
print(list)


def to_binary(n):
    if n == 0:
        return "0"
    elif n ==1:
        return '1'
    else: 
       return  to_binary(n//2) + str(n % 2)


print(to_binary(348))