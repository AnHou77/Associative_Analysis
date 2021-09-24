import os
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import time

DEBUG_MODE = 1

folder_path = './simulate_data2/'

purchase_data = pd.read_csv(folder_path+'purchase_data.csv')
product_data = pd.read_csv(folder_path+'product_data.csv',encoding='big5')
test_purchase = purchase_data.drop(columns=['MEMBER_ID'])
print(purchase_data)
# print(test_purchase.sample(frac=.7))
# test_purchase = test_purchase.sample(frac=.2)

if os.path.isfile(folder_path+'itemsets.csv'):
    print('itemsets.csv exists.')
    result = pd.read_csv(folder_path+'itemsets.csv')
else:
    print('itemsets.csv doesn\'t exists.')
    lasttime = time.time()
    result = apriori(test_purchase,min_support=0.05,use_colnames=True, verbose=1)
    result['itemsets'] = result['itemsets'].apply(lambda x: list(x)).astype("unicode")
    result.to_csv(folder_path+'itemsets.csv', index=False)
    newtime = time.time()
    print('It costs (seconds): ', newtime-lasttime)

result['itemsets'] = result['itemsets'].apply(lambda x: frozenset(x.replace('\'',"").strip('][').split(', ')))

if DEBUG_MODE:
    print(result)

if os.path.isfile(folder_path+'matrix.csv'):
    print('matrix.csv exists.')
    matrix = pd.read_csv(folder_path+'matrix.csv')
else:
    print('matrix.csv doesn\'t exists.')
    lasttime = time.time()
    matrix = association_rules(result, metric='support', min_threshold=0)
    matrix['consequent_len'] = matrix['consequents'].apply(lambda x: len(x))
    matrix['antecedents'] = matrix['antecedents'].apply(lambda x: list(x)).astype("unicode")
    matrix['consequents'] = matrix['consequents'].apply(lambda x: list(x)).astype("unicode")
    matrix.to_csv(folder_path+'matrix.csv', index=False)
    newtime = time.time()
    print('It costs (seconds): ', newtime-lasttime)

lasttime = time.time()
matrix['antecedents'] = matrix['antecedents'].apply(lambda x: frozenset(x.replace('\'',"").strip('][').split(', ')))
matrix['consequents'] = matrix['consequents'].apply(lambda x: frozenset(x.replace('\'',"").strip('][').split(', ')))
newtime = time.time()
print('It costs (seconds): ', newtime-lasttime)

id = input('Product ID: ')
product_matrix = matrix[(matrix['antecedents'] == {str(id)}) & (matrix['consequent_len'] < 2)]
print(product_matrix)
sorted_support = list(product_matrix.sort_values(by = ['support'], ascending=False)['consequents'].apply(lambda x: set(x).pop()))
sorted_confidence = list(product_matrix.sort_values(by = ['confidence'], ascending=False)['consequents'].apply(lambda x: set(x).pop()))
sorted_lift = list(product_matrix.sort_values(by = ['lift'], ascending=False)['consequents'].apply(lambda x: set(x).pop()))
# print(product_matrix.sort_values(by = ['support'], ascending=False))
print('Sorted by support: ',sorted_support)
print('Sorted by confidence: ',sorted_confidence)
print('Sorted by lift: ',sorted_lift)