# so this is a trial to see if I can still do things 
import numpy as np 

from PyPCAlg.pc_algorithm import run_pc_algorithm, field_pc_cpdag, \
    field_separation_sets
from PyPCAlg.examples.graph_4 import generate_data
from PyPCAlg.examples.graph_4 import oracle_indep_test
from PyPCAlg.examples.graph_4 import oracle_cond_indep_test
from PyPCAlg.examples.graph_4 import get_adjacency_matrix
print("hello world")

for i in range(1,2):
    print("aaa")
 # molto bene esisto ancora 



df = generate_data(sample_size=10)
independence_test_func = oracle_indep_test()
conditional_independence_test_func = oracle_cond_indep_test()

dic = run_pc_algorithm(
    data=df,
    indep_test_func=independence_test_func,
    cond_indep_test_func=conditional_independence_test_func,
    level=0.05
)
cpdag = dic[field_pc_cpdag]
separation_sets = dic[field_separation_sets]

print(f'The true causal graph is \n{get_adjacency_matrix()}')
print(f'\nThe CPDAG retrieved by PC is \n{cpdag}')

columns_to_extract = [2,4]
matrix = np.random.randint(0, 101, size=(5, 5))
tau = matrix[columns_to_extract, columns_to_extract]
print(matrix, tau)