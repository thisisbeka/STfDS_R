#generate random uniform data for test the solutions

import numpy as np
import pandas as pd
  

x = np.random.uniform(-50, 50, 500000)

df = pd.DataFrame({"x": x})

df.to_csv('data.csv', index = False)
