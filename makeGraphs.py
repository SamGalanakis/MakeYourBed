import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("darkgrid")

augmented = pd.read_csv("models\model_augmented\model_augmented_log.txt")
base  = pd.read_csv("models\model_5\model_5_log.txt")

plt.plot(augmented["Epoch"],augmented["TrainAccuracy"],label="Train")
plt.plot(augmented["Epoch"],augmented["TestAccuracy"],label="Test")
plt.legend()
plt.title("With Image Augmentation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()
plt.title("Without Image Augmentation")
plt.plot(base["Epoch"],base["TrainAccuracy"],label="Augmented")
plt.plot(base["Epoch"],base["TestAccuracy"],label="Test")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()
print("done")