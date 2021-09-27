import os
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import time

class Apriori:
    def __init__(self, config=None):
        self.data_dir_path = None
        self.purchase_data_path = None
        self.itemsets_data_path = None
        self.association_rule_matrix_path = None
        self.support_matrix_path = None
        self.confidence_matrix_path = None
        self.lift_matrix_path = None

        if config is not None:
            self.data_dir_path = config['dir_path']
            self.purchase_data_path = self.data_dir_path + config['purchase_data_path']
            self.itemsets_data_path = self.data_dir_path + config['itemsets_data_path']
            self.association_rule_matrix_path = self.data_dir_path + config['association_rule_matrix_path']
            self.support_matrix_path = self.data_dir_path + config['support_matrix_path']
            self.confidence_matrix_path = self.data_dir_path + config['confidence_matrix_path']
            self.lift_matrix_path = self.data_dir_path + config['lift_matrix_path']

        # get purchase data
        self.get_purchase_data()

    def get_purchase_data(self):
        try:
            self.purchase_data = pd.read_csv(self.purchase_data_path).drop(columns=['MEMBER_ID'])
        except Exception as e:
            print(e)

    def get_itemsets_data(self, DEBUG_MODE=True):
        if os.path.isfile(self.itemsets_data_path):
            if DEBUG_MODE:
                print('itemsets_data exists.')
            itemsets_data = pd.read_csv(self.itemsets_data_path)
            return itemsets_data
        else:
            if DEBUG_MODE:
                print('itemsets data doesn\'t exists.')
            return None
    
    def get_association_rule_matrix(self, DEBUG_MODE=True):
        if os.path.isfile(self.association_rule_matrix_path):
            if DEBUG_MODE:
                print('association_rule matrix exists.')
            association_rule_matrix = pd.read_csv(self.association_rule_matrix_path)
            return association_rule_matrix
        else:
            if DEBUG_MODE:
                print('association_rule matrix doesn\'t exists.')
            return None

    def get_support_matrix(self, DEBUG_MODE=True):
        if os.path.isfile(self.support_matrix_path):
            if DEBUG_MODE:
                print('support matrix exists.')
            support_matrix = pd.read_csv(self.support_matrix_path)
            return support_matrix
        else:
            if DEBUG_MODE:
                print('support matrix doesn\'t exists.')
            return None
            
    def get_confidence_matrix(self, DEBUG_MODE=True):
        if os.path.isfile(self.confidence_matrix_path):
            if DEBUG_MODE:
                print('confidence matrix exists.')
            confidence_matrix = pd.read_csv(self.confidence_matrix_path)
            return confidence_matrix
        else:
            if DEBUG_MODE:
                print('confidence matrix doesn\'t exists.')
            return None

    def get_lift_matrix(self, DEBUG_MODE=True):
        if os.path.isfile(self.lift_matrix_path):
            if DEBUG_MODE:
                print('lift matrix exists.')
            lift_matrix = pd.read_csv(self.lift_matrix_path)
            return lift_matrix
        else:
            if DEBUG_MODE:
                print('lift matrix doesn\'t exists.')
            return None


    def analysis(self, id: int, min_support=0.05,DEBUG_MODE=True):
        itemsets_data = self.get_itemsets_data(DEBUG_MODE=DEBUG_MODE)
        if itemsets_data is None:
            lasttime = time.time()
            itemsets_data = apriori(self.purchase_data,min_support=min_support,use_colnames=True, verbose=1)
            itemsets_data['itemsets'] = itemsets_data['itemsets'].apply(lambda x: list(x)).astype("unicode")
            itemsets_data.to_csv(self.itemsets_data_path, index=False)
            newtime = time.time()
            print('Generating itemsets data costs (seconds): ', newtime-lasttime)

        itemsets_data['itemsets'] = itemsets_data['itemsets'].apply(lambda x: frozenset(x.replace('\'',"").strip('][').split(', ')))

        if DEBUG_MODE:
            print('itemsets data:')
            print(itemsets_data)
        
        association_rule_matrix = self.get_association_rule_matrix(DEBUG_MODE=DEBUG_MODE)
        if association_rule_matrix is None:
            lasttime = time.time()
            association_rule_matrix = association_rules(itemsets_data, metric='support', min_threshold=0)
            association_rule_matrix['antecedent_len'] = association_rule_matrix['antecedents'].apply(lambda x: len(x))
            association_rule_matrix['consequent_len'] = association_rule_matrix['consequents'].apply(lambda x: len(x))
            association_rule_matrix['antecedents'] = association_rule_matrix['antecedents'].apply(lambda x: list(x)).astype("unicode")
            association_rule_matrix['consequents'] = association_rule_matrix['consequents'].apply(lambda x: list(x)).astype("unicode")
            association_rule_matrix.to_csv(self.association_rule_matrix_path, index=False)
            newtime = time.time()
            print('Generating association rule matrix costs (seconds): ', newtime-lasttime)

        association_rule_matrix['antecedents'] = association_rule_matrix['antecedents'].apply(lambda x: frozenset(x.replace('\'',"").strip('][').split(', ')))
        association_rule_matrix['consequents'] = association_rule_matrix['consequents'].apply(lambda x: frozenset(x.replace('\'',"").strip('][').split(', ')))

        support_matrix = self.get_support_matrix(DEBUG_MODE=DEBUG_MODE)
        confidence_matrix = self.get_confidence_matrix(DEBUG_MODE=DEBUG_MODE)
        lift_matrix = self.get_lift_matrix(DEBUG_MODE=DEBUG_MODE)

        if (support_matrix is None) or (confidence_matrix is None) or (lift_matrix is None):
            products_size = len(self.purchase_data.columns)
            support_matrix = np.zeros((products_size,products_size))
            confidence_matrix = np.zeros((products_size,products_size))
            lift_matrix = np.zeros((products_size,products_size))

            temp_matrix = association_rule_matrix[(association_rule_matrix['antecedent_len'] < 2) & (association_rule_matrix['consequent_len'] < 2)]

            old_time = time.time()
            for i in range(products_size):
                for j in range(products_size):
                    if i == j:
                        support_matrix[i][j] = temp_matrix[temp_matrix['antecedents'] == {str(i)}].iloc[0]['antecedent support']
                        confidence_matrix[i][j] = None
                        lift_matrix[i][j] = None
                    else:
                        try:
                            support_matrix[i][j] = temp_matrix[(temp_matrix['antecedents'] == {str(i)}) & (temp_matrix['consequents'] == {str(j)})].iloc[0]['support']
                        except:
                            support_matrix[i][j] = None
                        try:
                            confidence_matrix[i][j] = temp_matrix[(temp_matrix['antecedents'] == {str(i)}) & (temp_matrix['consequents'] == {str(j)})].iloc[0]['confidence']
                        except:
                            confidence_matrix[i][j] = None
                        try:
                            lift_matrix[i][j] = temp_matrix[(temp_matrix['antecedents'] == {str(i)}) & (temp_matrix['consequents'] == {str(j)})].iloc[0]['lift']
                        except:
                            lift_matrix[i][j] = None
            new_time = time.time()
            print('Generating support, confidence, lift matrix costs (seconds): ', new_time-old_time)

            support_matrix = pd.DataFrame(support_matrix)
            confidence_matrix = pd.DataFrame(confidence_matrix)
            lift_matrix = pd.DataFrame(lift_matrix)

            support_matrix.to_csv(self.support_matrix_path, index=False)
            confidence_matrix.to_csv(self.confidence_matrix_path, index=False)
            lift_matrix.to_csv(self.lift_matrix_path, index=False)

        if DEBUG_MODE:
            print('ITEM-ITEM Support Matrix: ', support_matrix)
            print('ITEM-ITEM Confidence Matrix', confidence_matrix)
            print('ITEM-ITEM Lift Matrix', lift_matrix)


        # get sorted products with id
        matrix_sorted_by_id = association_rule_matrix[(association_rule_matrix['antecedents'] == {str(id)}) & (association_rule_matrix['consequent_len'] < 2)]

        if DEBUG_MODE:
            print('association rule matrix sorted by id: ')
            print(matrix_sorted_by_id)

        sorted_support = list(matrix_sorted_by_id.sort_values(by = ['support'], ascending=False)['consequents'].apply(lambda x: set(x).pop()))
        sorted_confidence = list(matrix_sorted_by_id.sort_values(by = ['confidence'], ascending=False)['consequents'].apply(lambda x: set(x).pop()))
        sorted_lift = list(matrix_sorted_by_id.sort_values(by = ['lift'], ascending=False)['consequents'].apply(lambda x: set(x).pop()))

        print('Related products sorted by support: ',sorted_support)
        print('Related products sorted by confidence: ',sorted_confidence)
        print('Related products sorted by lift: ',sorted_lift)

        return sorted_support, sorted_confidence, sorted_lift