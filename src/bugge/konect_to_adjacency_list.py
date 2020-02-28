import sys

input_file = open(sys.argv[1], 'r')
output_file = open(sys.argv[2], 'w')

old_lines = input_file.readlines()
old_lines = old_lines[2:]
old_lines = [line.split(' ') for line in old_lines]

new_lines = {}
for old_line in old_lines:
    node = old_line[0]
    if node in new_lines:
        new_lines[node] = new_lines[node] + ' ' + old_line[1]
    else:
        new_lines[node] = node + ' ' + old_line[1]

for node, new_line in new_lines.items():
    output_file.write(new_line + '\n')

input_file.close()
output_file.close()
