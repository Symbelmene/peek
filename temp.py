

# 4 digit combination

# Each digit must be greater than 2

# Each digit must be less than 9
# The first digit must be greater than 4

# The sum of all digits must be greater than 15 and less than 25

# How many combinations are there?

def combination_finder():
    combinations = []
    for i in range(4, 9):  # Plus
        for j in range(2, 9):   # Minus
            for k in range(3, 6):  # Multiply
                for l in range(2, 6):  # Divide
                    if i + j + k + l > 16 and i + j + k + l < 22:
                        combinations.append([i, j, k, l])
    return combinations


if __name__ == '__main__':
    combs = combination_finder()
    print(len(combs))
    for comb in combs:
        print(comb)