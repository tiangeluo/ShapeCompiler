left = []
right = []
for i in range(len(word_map)):
    if (tokenizer.decode(tokenizer.encode(list(word_map.keys())[i])).strip() == list(word_map.keys())[i]) == False:
        left.append(tokenizer.decode(tokenizer.encode(list(word_map.keys())[i])).strip())
        right.append(list(word_map.keys())[i])
        
    
with open('left_tokenizer.txt', 'w') as f:
    for i in left:
        f.writelines(i+'\n')
with open('right_tokenizer.txt', 'w') as f:
    for i in right:
        f.writelines(i+'\n')