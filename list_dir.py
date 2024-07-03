import os

directory_path = 'E:/Ford/CP_Match/prototype/mmaction2/data/lotus_not_fix/train/'

file_list = os.listdir(directory_path)

with open('file_list.txt', 'w') as file:
    for file_name in file_list:
        file.write(f"{file_name}\n")
