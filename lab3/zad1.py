# import numpy as np
# import pandas as pd
# import matplotlib as plt 
# import seaborn as sns

# df = pd.read_csv("StudentPerformanceFactors.csv",sep=",")

# df.head()
# df["Parental_Involvement"] = df["Parental_Involvement"].replace(
#     {"Low":1,
#      "Medium":2,
#      "High":3
#     }
# )
# df["Access_to_Resources"] = df["Access_to_Resources"].replace(
#     {"Low":1,
#      "Medium":2,
#      "High":3
#     }
# )
# df.head()

# correlationMatrix = df.select_dtypes(include=[np.number]).corr()
# sns.heatmap(correlationMatrix, annot=True, cmap="coolwarm", fmt=".2f", square=True, linewidths=2)
# plt.title('Macierz koleracji')
# plt.show()
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
x = np.arange(-3,3, 0.1).reshape((-1,1))
y = np.tanh(x) + np.random.randn(*x.shape)*0.2
ypred = LinearRegression().fit(x,y).predict(x)
plt.scatter(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x, ypred)
plt.legend([ 'F(x) - aproksymująca',
 'f(x) - aproksymowana zaszumiona'])