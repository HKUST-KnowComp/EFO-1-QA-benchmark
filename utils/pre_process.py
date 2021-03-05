def preprocess(input_file, output_file):
    all_queries = []
    with open(input_file, 'r', errors='ignore') as infile:
        for line in infile.readlines():
            e1, r, e2 = line.strip().split('\t')
            r_num = len(r.split('.'))
            if r_num == 1:
                all_queries.append([e1, r, e2])
            elif r_num == 2:
                r1, r2 = r.split('.')
                all_queries.append([e1, r1, e2])
                all_queries.append([e1, r2, e2])
            else:
                print(r_num, r)
    with open(output_file, 'w') as outfile:
        for query in all_queries:
            e1, r, e2 = query
            outfile.write(e1 + '\t' + r + '\t' + e2 + '\n')


preprocess('../datasets_knowledge_embedding/FB15k-237/train.txt', '../datasets_knowledge_embedding/FB15k-237/my_train.txt')
preprocess('../datasets_knowledge_embedding/FB15k-237/valid.txt', '../datasets_knowledge_embedding/FB15k-237/my_valid.txt')
preprocess('../datasets_knowledge_embedding/FB15k-237/test.txt', '../datasets_knowledge_embedding/FB15k-237/my_test.txt')


