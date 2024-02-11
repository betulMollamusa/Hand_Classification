import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder,StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay

# Veriyi al
data = pd.read_csv("otu.csv",dtype='unicode')

# Veriyi ters çevir
dataT = data.transpose()

#veriyi normalize et
y=dataT.iloc[:,0].values
Y=pd.DataFrame(y)

x=dataT.loc[:,dataT.columns != 0].values
XD=pd.DataFrame(x)
normalData = StandardScaler()
normalData.fit(XD)
normalized_data = normalData.transform(XD)
X = pd.DataFrame(normalized_data)

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Özellik seçimi
selector = SelectKBest(f_classif, k=3302)
X_train_selected = selector.fit_transform(X_train, Y_train)
X_test_selected = selector.transform(X_test)

# Sınıflandırıcıyı eğit
classifier = KNeighborsClassifier()
classifier.fit(X_train_selected, Y_train)

# Çapraz doğrulama ile sınıflandırıcıyı değerlendir
cv_scores = cross_val_score(classifier, X_train_selected, Y_train, cv=10)  # 10 katlı çapraz doğrulama

# Test setini kullanarak sınıflandırma yap
Y_pred = classifier.predict(X_test_selected)

# Performans ölçümlerini hesapla
accuracy = accuracy_score(Y_test, Y_pred)
report = classification_report(Y_test, Y_pred)

# AUC için string-float ayarlaması yapılır
label_encoder = LabelEncoder()
# Gerçek etiketleri (Y_test) sayısal hale getir
y_true_numeric = label_encoder.fit_transform(Y_test)
# Tahmin edilen etiketleri (Y_pred) sayısal hale getir
y_pred_numeric = label_encoder.transform(Y_pred)
auc = roc_auc_score(y_true_numeric, y_pred_numeric)

# CEVAP-CROSS-VALIDATION (Çapraz doğrulama skorları)
print("Cross-Validation Scores:", cv_scores)
print("Mean Cross-Validation Score:", cv_scores.mean())

# DOĞRULUK VE İSTATİSTİK RAPORU
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")

# CEVAP-AUC
print(f"AUC: {auc}")

# Confusion Matrix
conf_mat = confusion_matrix(Y_test, Y_pred)

sensitivity = conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[0, 1])  # Sensitivity (True Positive Rate)
specificity = conf_mat[1, 1] / (conf_mat[1, 0] + conf_mat[1, 1])  # Specificity (True Negative Rate)

# CEVAP-SENSITIVITY,SPECIFICITY
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)

# ROC Curve (AUC)
roc_display = RocCurveDisplay.from_estimator(classifier, X_test_selected, Y_test)
roc_display.plot()
plt.title("ROC Curve")
plt.show()

# Precision-Recall Curve
pr_display = PrecisionRecallDisplay.from_estimator(classifier, X_test_selected, Y_test)
pr_display.plot()
plt.title("Precision-Recall Curve")
plt.show()
