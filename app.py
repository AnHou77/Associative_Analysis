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
    sorted_support, sorted_confidence, sorted_lift = apriori_model.analysis(id,DEBUG_MODE=True)

if __name__ == "__main__":
    main()