with open("input.txt") as file:
    largest_line = ""
    for line in file:
        if len(line) > len(largest_line):
            largest_line = line
print("The largest line is:", len(largest_line))
