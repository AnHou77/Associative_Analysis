# Associative_Analysis

## Installation
```
pip install -r requirements.txt
```

## Model Introduction

### `class Apriori(config)`

#### `__init__(self, config=None)`
initial the apriori model 

##### Parameters
- **config** (`json`) - setting dir path, data path.

#### `analysis(self, id: int, min_support=0.05,DEBUG_MODE=True)`
associative rule analysis

##### Parameters
- **id** (`int`) - Use this product id to get a sorted related products list with support, confidence, lift.
- **min_support** (`float`, defaults to `0.05`) - Filter out itemsets below this support value.
- **DEBUG_MODE** (`bool`, defaults to `True`) - Present some detailed information in the analysis.

##### Returns
- **sorted_support** (`list`) - Product ids associated with the given id sorted by support value
- **sorted_confidence** (`list`) - Product ids associated with the given id sorted by confidence value
- **sorted_lift** (`list`) - Product ids associated with the given id sorted by lift value

### **Tool function usage**
#### Overview
#### `get_purchase_data(self, DEBUG_MODE=True)`
- read purchase data from `purchase_data_path` and drop column [`MEMBER_ID`]
#### `get_itemsets_data(self, DEBUG_MODE=True)`
- check if `itemsets_data_path` file exists, read itemsets data and return `itemsets_data`.
- if `itemsets_data_path` file does't exist, return `None`, in `analysis()` will generate new itemsets data.
#### `get_association_rule_matrix(self, DEBUG_MODE=True)`
- check if `association_rule_matrix_path` file exists, read association rule matrix and return `association_rule_matrix`.
- if `association_rule_matrix_path` file does't exist, return `None`, in `analysis()` will generate new association rule matrix.
#### `get_support_matrix(self, DEBUG_MODE=True)`
- check if `support_matrix_path` file exists, read **ITEM-ITEM Support Matrix** and return `support_matrix`.
- if `support_matrix_path` file does't exist, return `None`, in `analysis()` will generate new support matrix.
#### `get_confidence_matrix(self, DEBUG_MODE=True)`
- check if `confidence_matrix_path` file exists, read **ITEM-ITEM Confidence Matrix** and return `confidence_matrix`.
- if `confidence_matrix_path` file does't exist, return `None`, in `analysis()` will generate new confidence matrix.
#### `get_lift_matrix(self, DEBUG_MODE=True)`
- check if `lift_matrix_path` file exists, read **ITEM-ITEM Lift Matrix** and return `lift_matrix`.
- if `lift_matrix_path` file does't exist, return `None`, in `analysis()` will generate new lift matrix.

#### Reason
With these function, in the case that the source data hasn't changed, we can save a lot of calculation time.

## Model usage example
### source code
```=python
import pandas as pd
import numpy as np
import json
from apriori import Apriori

def main():
    with open('config.json', 'r') as f:
        config = json.load(f)
        f.close()
    apriori_model = Apriori(config=config)
    id = int(input('Product ID: '))
    sorted_support, sorted_confidence, sorted_lift = apriori_model.analysis(id, min_support=0.05, DEBUG_MODE=False)

if __name__ == "__main__":
    main()
```
### Output (DEBUG_MODE = False)
```
Product ID: 0
Related products sorted by support:  ['50', '51', '31', '2', '52', '15', '12', '27', '14', '39', '30', '54', '35', '17', '47', '16', '33', '22', '37', '3', '13', '44', '24', '20', '40', '9', '7', '48', '18', '26', '61', '49', '41', '42', '1', '4', '34', '63', '38', '53', '23', '19', '6', '45', '25', '5', '11', '32', '10', '29', '28', '55', '46', '36', '21', '43', '8', '62', '57', '64', '58', '60', '59', '65']
Related products sorted by confidence:  ['50', '51', '31', '2', '52', '15', '12', '27', '14', '39', '30', '54', '35', '17', '47', '16', '33', '22', '37', '3', '13', '44', '24', '20', '40', '9', '7', '48', '18', '26', '61', '49', '41', '42', '1', '4', '34', '63', '38', '53', '23', '19', '6', '45', '25', '5', '11', '32', '10', '29', '28', '55', '46', '36', '21', '43', '8', '62', '57', '64', '58', '60', '59', '65']
Related products sorted by lift:  ['12', '23', '42', '11', '34', '14', '46', '63', '9', '17', '24', '38', '41', '49', '61', '45', '31', '22', '15', '53', '50', '27', '51', '16', '13', '2', '52', '35', '55', '26', '10', '40', '19', '6', '7', '18', '21', '30', '43', '3', '39', '62', '54', '64', '1', '36', '25', '37', '47', '44', '20', '29', '48', '33', '32', '65', '60', '57', '8', '4', '5', '28', '58', '59']
```

### Output (DEBUG_MODE = True)
```
Product ID: 0
itemsets_data exists.
itemsets data:
        support                      itemsets
0      0.455840                           (0)
1      0.358974                           (1)
2      0.552707                           (2)
3      0.393162                           (3)
4      0.421652                           (4)
...         ...                           ...
89907  0.056980  (30, 31, 51, 39, 38, 50, 48)
89908  0.056980  (30, 31, 51, 39, 50, 48, 47)
89909  0.054131  (30, 31, 51, 39, 50, 48, 49)
89910  0.054131  (31, 51, 39, 38, 50, 48, 49)
89911  0.054131  (31, 51, 39, 50, 48, 47, 49)

[89912 rows x 2 columns]
association_rule matrix exists.
support matrix doesn't exists.
confidence matrix doesn't exists.
lift matrix doesn't exists.
Generating support, confidence, lift matrix costs (seconds):  12.022193670272827
ITEM-ITEM Support Matrix:            0         1         2         3         4         5         6         7         8         9         10  ...        55        56        57        58        59        60        61        62        63   
     64        65
0   0.455840  0.162393  0.273504  0.182336  0.162393  0.133903  0.145299  0.176638  0.088319  0.176638  0.122507  ...  0.102564       NaN  0.068376  0.065527  0.056980  0.062678  0.170940  0.076923  0.156695  0.068376  0.054131
1   0.162393  0.358974  0.225071  0.150997  0.142450  0.133903  0.085470  0.122507  0.102564  0.122507  0.111111  ...  0.088319       NaN  0.056980  0.062678  0.054131  0.059829  0.094017  0.068376  0.128205  0.056980  0.062678
2   0.273504  0.225071  0.552707  0.213675  0.253561  0.247863  0.148148  0.207977  0.099715  0.199430  0.159544  ...  0.133903  0.071225  0.099715  0.116809  0.105413  0.085470  0.188034  0.094017  0.196581  0.094017       NaN
3   0.182336  0.150997  0.213675  0.393162  0.233618  0.148148  0.173789  0.190883  0.099715  0.142450  0.148148  ...  0.079772       NaN  0.051282  0.074074  0.051282  0.056980  0.096866  0.054131  0.113960  0.059829       NaN
4   0.162393  0.142450  0.253561  0.233618  0.421652  0.168091  0.162393  0.182336  0.099715  0.136752  0.159544  ...  0.088319  0.054131       NaN  0.068376  0.062678  0.062678  0.102564  0.054131  0.116809       NaN  0.065527
..       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...  ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...
61  0.170940  0.094017  0.188034  0.096866  0.102564  0.125356  0.102564  0.122507  0.074074  0.113960  0.068376  ...  0.085470       NaN  0.074074  0.062678       NaN  0.076923  0.321937       NaN  0.119658  0.059829       NaN
62  0.076923  0.068376  0.094017  0.054131  0.054131       NaN       NaN       NaN       NaN       NaN       NaN  ...       NaN       NaN       NaN       NaN       NaN       NaN       NaN  0.168091       NaN       NaN       NaN
63  0.156695  0.128205  0.196581  0.113960  0.116809  0.128205  0.076923  0.094017  0.054131  0.096866  0.062678  ...  0.091168       NaN       NaN  0.056980       NaN  0.065527  0.119658       NaN  0.284900  0.059829       NaN
64  0.068376  0.056980  0.094017  0.059829       NaN  0.076923       NaN  0.051282       NaN  0.056980       NaN  ...  0.059829       NaN       NaN       NaN       NaN       NaN  0.059829       NaN  0.059829  0.150997       NaN
65  0.054131  0.062678       NaN       NaN  0.065527  0.062678       NaN  0.056980       NaN       NaN  0.056980  ...       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN  0.131054

[66 rows x 66 columns]
ITEM-ITEM Confidence Matrix           0         1         2         3         4         5         6         7         8         9         10  ...        55        56        57        58        59        60        61        62        63  
      64        65
0        NaN  0.356250  0.600000  0.400000  0.356250  0.293750  0.318750  0.387500  0.193750  0.387500  0.268750  ...  0.225000       NaN  0.150000  0.143750  0.125000  0.137500  0.375000  0.168750  0.343750  0.150000  0.118750
1   0.452381       NaN  0.626984  0.420635  0.396825  0.373016  0.238095  0.341270  0.285714  0.341270  0.309524  ...  0.246032       NaN  0.158730  0.174603  0.150794  0.166667  0.261905  0.190476  0.357143  0.158730  0.174603
2   0.494845  0.407216       NaN  0.386598  0.458763  0.448454  0.268041  0.376289  0.180412  0.360825  0.288660  ...  0.242268  0.128866  0.180412  0.211340  0.190722  0.154639  0.340206  0.170103  0.355670  0.170103       NaN
3   0.463768  0.384058  0.543478       NaN  0.594203  0.376812  0.442029  0.485507  0.253623  0.362319  0.376812  ...  0.202899       NaN  0.130435  0.188406  0.130435  0.144928  0.246377  0.137681  0.289855  0.152174       NaN
4   0.385135  0.337838  0.601351  0.554054       NaN  0.398649  0.385135  0.432432  0.236486  0.324324  0.378378  ...  0.209459  0.128378       NaN  0.162162  0.148649  0.148649  0.243243  0.128378  0.277027       NaN  0.155405
..       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...  ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...
61  0.530973  0.292035  0.584071  0.300885  0.318584  0.389381  0.318584  0.380531  0.230088  0.353982  0.212389  ...  0.265487       NaN  0.230088  0.194690       NaN  0.238938       NaN       NaN  0.371681  0.185841       NaN
62  0.457627  0.406780  0.559322  0.322034  0.322034       NaN       NaN       NaN       NaN       NaN       NaN  ...       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN
63  0.550000  0.450000  0.690000  0.400000  0.410000  0.450000  0.270000  0.330000  0.190000  0.340000  0.220000  ...  0.320000       NaN       NaN  0.200000       NaN  0.230000  0.420000       NaN       NaN  0.210000       NaN
64  0.452830  0.377358  0.622642  0.396226       NaN  0.509434       NaN  0.339623       NaN  0.377358       NaN  ...  0.396226       NaN       NaN       NaN       NaN       NaN  0.396226       NaN  0.396226       NaN       NaN
65  0.413043  0.478261       NaN       NaN  0.500000  0.478261       NaN  0.434783       NaN       NaN  0.434783  ...       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN

[66 rows x 66 columns]
ITEM-ITEM Lift Matrix           0         1         2         3         4         5         6         7         8         9         10  ...        55        56        57        58        59        60        61        62        63        
64        65
0        NaN  0.992411  1.085567  1.017391  0.844890  0.838262  1.045619  1.038263  0.883198  1.203650  1.071946  ...  1.081849       NaN  0.892373  0.813810  0.756466  0.893750  1.164823  1.003919  1.206562  0.993396  0.906114
1   0.992411       NaN  1.134389  1.069876  0.941120  1.064460  0.781041  0.914395  1.302412  1.060051  1.234578  ...  1.182975       NaN  0.944310  0.988479  0.912562  1.083333  0.813527  1.133172  1.253571  1.051213  1.332298
2   1.085567  1.134389       NaN  0.983303  1.088012  1.279733  0.879275  1.008224  0.822399  1.120792  1.151359  ...  1.164878  1.330352  1.073301  1.196458  1.154195  1.005155  1.056747  1.011969  1.248402  1.126532       NaN
3   1.017391  1.069876  0.983303       NaN  1.409224  1.075292  1.450020  1.300863  1.156126  1.125433  1.502964  ...  0.975581       NaN  0.775976  1.066620  0.789355  0.942029  0.765294  0.819086  1.017391  1.007793       NaN
4   0.844890  0.941120  1.088012  1.409224       NaN  1.137607  1.263387  1.158655  1.078010  1.007414  1.509214  ...  1.007127  1.325318       NaN  0.918047  0.899581  0.966216  0.755561  0.763743  0.972365       NaN  1.185811
..       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...  ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...
61  1.164823  0.813527  1.056747  0.765294  0.755561  1.111159  1.045075  1.019591  1.048845  1.099538  0.847144  ...  1.276518       NaN  1.368832  1.102198       NaN  1.553097       NaN       NaN  1.304602  1.230756       NaN
62  1.003919  1.133172  1.011969  0.819086  0.763743       NaN       NaN       NaN       NaN       NaN       NaN  ...       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN
63  1.206562  1.253571  1.248402  1.017391  0.972365  1.284146  0.885701  0.884198  0.866104  1.056106  0.877500  ...  1.538630       NaN       NaN  1.132258       NaN  1.495000  1.304602       NaN       NaN  1.390755       NaN
64  0.993396  1.051213  1.126532  1.007793       NaN  1.453751       NaN  0.909981       NaN  1.172149       NaN  ...  1.905143       NaN       NaN       NaN       NaN       NaN  1.230756       NaN  1.390755       NaN       NaN
65  0.906114  1.332298       NaN       NaN  1.185811  1.364793       NaN  1.164952       NaN       NaN  1.734190  ...       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN

[66 rows x 66 columns]
association rule matrix sorted by id: 
    antecedents consequents  antecedent support  consequent support   support  confidence      lift  leverage  conviction  antecedent_len  consequent_len
1           (0)         (1)             0.45584            0.358974  0.162393     0.35625  0.992411 -0.001242    0.995768               1               1
3           (0)         (2)             0.45584            0.552707  0.273504     0.60000  1.085567  0.021558    1.118234               1               1
4           (0)         (3)             0.45584            0.393162  0.182336     0.40000  1.017391  0.003117    1.011396               1               1
7           (0)         (4)             0.45584            0.421652  0.162393     0.35625  0.844890 -0.029813    0.898404               1               1
9           (0)         (5)             0.45584            0.350427  0.133903     0.29375  0.838262 -0.025836    0.919749               1               1
..          ...         ...                 ...                 ...       ...         ...       ...       ...         ...             ...             ...
119         (0)        (61)             0.45584            0.321937  0.170940     0.37500  1.164823  0.024188    1.084900               1               1
120         (0)        (62)             0.45584            0.168091  0.076923     0.16875  1.003919  0.000300    1.000793               1               1
123         (0)        (63)             0.45584            0.284900  0.156695     0.34375  1.206562  0.026826    1.089676               1               1
125         (0)        (64)             0.45584            0.150997  0.068376     0.15000  0.993396 -0.000455    0.998827               1               1
126         (0)        (65)             0.45584            0.131054  0.054131     0.11875  0.906114 -0.005609    0.986038               1               1

[64 rows x 11 columns]
Related products sorted by support:  ['50', '51', '31', '2', '52', '15', '12', '27', '14', '39', '30', '54', '35', '17', '47', '16', '33', '22', '37', '3', '13', '44', '24', '20', '40', '9', '7', '48', '18', '26', '61', '49', '41', '42', '1', '4', '34', '63', '38', '53', '23', '19', '6', '45', '25', '5', '11', '32', '10', '29', '28', '55', '46', '36', '21', '43', '8', '62', '57', '64', '58', '60', '59', '65']
Related products sorted by confidence:  ['50', '51', '31', '2', '52', '15', '12', '27', '14', '39', '30', '54', '35', '17', '47', '16', '33', '22', '37', '3', '13', '44', '24', '20', '40', '9', '7', '48', '18', '26', '61', '49', '41', '42', '1', '4', '34', '63', '38', '53', '23', '19', '6', '45', '25', '5', '11', '32', '10', '29', '28', '55', '46', '36', '21', '43', '8', '62', '57', '64', '58', '60', '59', '65']
Related products sorted by lift:  ['12', '23', '42', '11', '34', '14', '46', '63', '9', '17', '24', '38', '41', '49', '61', '45', '31', '22', '15', '53', '50', '27', '51', '16', '13', '2', '52', '35', '55', '26', '10', '40', '19', '6', '7', '18', '21', '30', '43', '3', '39', '62', '54', '64', '1', '36', '25', '37', '47', '44', '20', '29', '48', '33', '32', '65', '60', '57', '8', '4', '5', '28', '58', '59']
```