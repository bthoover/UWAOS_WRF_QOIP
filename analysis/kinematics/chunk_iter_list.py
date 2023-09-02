###################################################
#
# chunk_list: Given a master list and a preferred
#             number of sections, chunk the list
#             into roughly-equal sized sections
#             and return a list containing all
#             sub-list sections.
# 
def chunk_list(master_list,n_sec):
    master_len = len(master_list)
    # Number of sections containing master_len//n_sec + 1 elements
    n_bump_sublist = master_len % n_sec
    bump_len = master_len//n_sec + 1
    # Number of sections containing master_len//n_sec elements
    n_base_sublist = n_sec - n_bump_sublist
    base_len = master_len//n_sec
    # Initialize sub_lists
    sub_lists=[]
    idx_beg=-9
    idx_end=-9
    idx_start=0
    # Chunk through bump-lists (sub-lists containing an additional element)
    for i in range(n_bump_sublist):
        idx_beg = 0 + i*bump_len
        idx_end = idx_beg + bump_len
        chunk = master_list[idx_beg:idx_end]
        sub_lists.append(chunk)
        idx_start = idx_end
    # chunk through base-lists (sub-lists containing base_len elements)
    for i in range(n_base_sublist):
        idx_beg = idx_start + i*base_len
        idx_end = idx_beg + base_len
        chunk = master_list[idx_beg:idx_end]
        sub_lists.append(chunk)
    return sub_lists


if __name__ == "__main__":
    import argparse
    import numpy as np
    # Define and collect input arguments (strings)
    parser = argparse.ArgumentParser(description='Define iteration list')
    parser.add_argument('iterBeg', metavar='iterBeg', type=int, help='beginning iteration in xx format')
    parser.add_argument('iterEnd', metavar='iterEnd', type=int, help='ending iteration in xx format')
    parser.add_argument('nChunks', metavar='nChunks', type=int, help='number of chunks to create')
    commandInputs = parser.parse_args()
    iterBeg = commandInputs.iterBeg
    iterEnd = commandInputs.iterEnd
    nChunks = commandInputs.nChunks
    # Construct total iteration list (integers)
    iterListInt = np.arange(start=iterBeg, stop=iterEnd + 1, dtype='int').tolist()
    # Construct total iteration list (strings)
    iterListStr = []
    for i in iterListInt:
        if i < 10:
            iterListStr.append('0' + str(i))
        else:
            iterListStr.append(str(i))
    # Generate n_chunks sub-lists from date_list
    iterListChunks=chunk_list(iterListStr, nChunks)
    # Print each sub-list as a single space-delimeted string
    for subList in iterListChunks:
        outputStr = ''
        for i in subList:
            outputStr = outputStr + i + ' '
        print(outputStr)
