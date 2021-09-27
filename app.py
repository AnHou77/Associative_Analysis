import os
from mlxtend.frequent_patterns import apriori
import pandas as pd
import numpy as np
import json
from apriori import Apriori
import time

def main():
    with open('config.json', 'r') as f:
        config = json.load(f)
        f.close()
    apriori_model = Apriori(config=config)
    id = int(input('Product ID: '))
    sorted_support, sorted_confidence, sorted_lift = apriori_model.analysis(id)

    # association_rule_matrix = apriori_model.get_association_rule_matrix()
    # association_rule_matrix['antecedents'] = association_rule_matrix['antecedents'].apply(lambda x: frozenset(x.replace('\'',"").strip('][').split(', ')))
    # association_rule_matrix['consequents'] = association_rule_matrix['consequents'].apply(lambda x: frozenset(x.replace('\'',"").strip('][').split(', ')))
    # temp = association_rule_matrix[(association_rule_matrix['antecedent_len'] < 2) & (association_rule_matrix['consequent_len'] < 2)]
    # print(temp)

    # support_matrix = np.zeros((65,65))
    # confidence_matrix = np.zeros((65,65))
    # lift_matrix = np.zeros((65,65))

    # old_time = time.time()
    # for i in range(65):
    #     for j in range(65):
    #         if i == j:
    #             support_matrix[i][j] = temp[temp['antecedents'] == {str(i)}].iloc[0]['antecedent support']
    #             confidence_matrix[i][j] = None
    #             lift_matrix[i][j] = None
    #         else:
    #             try:
    #                 support_matrix[i][j] = temp[(temp['antecedents'] == {str(i)}) & (temp['consequents'] == {str(j)})].iloc[0]['support']
    #             except:
    #                 support_matrix[i][j] = None
    #             try:
    #                 confidence_matrix[i][j] = temp[(temp['antecedents'] == {str(i)}) & (temp['consequents'] == {str(j)})].iloc[0]['confidence']
    #             except:
    #                 confidence_matrix[i][j] = None
    #             try:
    #                 lift_matrix[i][j] = temp[(temp['antecedents'] == {str(i)}) & (temp['consequents'] == {str(j)})].iloc[0]['lift']
    #             except:
    #                 lift_matrix[i][j] = None

    # new_time = time.time()
    # print('costs: ', new_time-old_time)

    # support_matrix = pd.DataFrame(support_matrix)
    # confidence_matrix = pd.DataFrame(confidence_matrix)
    # lift_matrix = pd.DataFrame(lift_matrix)

    # print(support_matrix)
    # print(confidence_matrix)
    # print(lift_matrix)

if __name__ == "__main__":
    main()