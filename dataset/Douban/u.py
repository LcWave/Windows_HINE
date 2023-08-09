def remove_nonexistent_nodes(file1_path, file2_path):
    # Read nodeids from file2 into another set (both nodeid1 and nodeid2)
    with open(file2_path, 'r') as f2:
        nodes_from_file2 = {line.split('\t')[0] for line in f2}
        nodes_from_file2 |= {line.split('\t')[1] for line in f2}

    # Remove nodeids from file1 if they are not in nodes_from_file2
    updated_lines = []
    with open(file1_path, 'r') as f1:
        for line in f1:
            node_id = line.split('\t')[0]
            node_id = node_id[1:]
            if node_id in nodes_from_file2:
                updated_lines.append(line)

    # Write the updated lines back to file1
    with open(file3_path, 'w') as f3:
        f3.writelines(updated_lines)

# Example usage:
file1_path = 'label.txt'
file2_path = 'edge.txt'
file3_path = 'label1.txt'
remove_nonexistent_nodes(file1_path, file2_path)
