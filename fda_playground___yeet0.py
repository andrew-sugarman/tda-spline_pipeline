import re
from pathlib import Path
import pandas as pd
import numpy as np
import skfda
from skfda.representation.basis import BSplineBasis
import matplotlib.pyplot as plt
import pickle

# load functional data object from spline fitting script
with open("fd_aggregated.pkl", "rb") as f:
    fd_aggregated = pickle.load(f)

print("Type:", type(fd_aggregated))
print("Summary of fd_aggregated:")
# print(fd_aggregated)

if hasattr(fd_aggregated, "coefficients"):
    print("Coefficients shape:", fd_aggregated.coefficients.shape)
if hasattr(fd_aggregated, "basis"):
    print("Basis details:", fd_aggregated.basis)

fd_aggregated.plot()
plt.title("B-Splines ")
plt.show()
