import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

PATIENTS_CSV_PATH = r"C:\Users\picul\Videos\Applied\Dataset_Full\patients.csv"
df = pd.read_csv(PATIENTS_CSV_PATH)

df['AHI'] = pd.to_numeric(df['AHI'], errors='coerce').fillna(0.0)

df[['BPsys', 'BPdia']] = df['BPsys/BPdia'].str.split('/', expand=True).astype(float)
df['sex'] = (df['sex'] == 'M').astype(int)

def ahi_class(v):
    if v < 5:  return 'Asymptomatic'
    if v < 30: return 'Moderate'
    return 'Severe'

df['AHI_Class'] = df['AHI'].apply(ahi_class)

features = ['age', 'sex', 'height', 'weight', 'pulse', 'BPsys', 'BPdia']
df[features] = df[features].apply(pd.to_numeric, errors='coerce')
df = df.dropna(subset=features)

X      = df[features]
y      = df['AHI_Class']
groups = df['user_id']  # both nights of the same patient stay together

clf = RandomForestClassifier(n_estimators=100, random_state=42)
cv  = StratifiedGroupKFold(n_splits=5)

y_pred = cross_val_predict(clf, X, y, cv=cv, groups=groups)

print(classification_report(y, y_pred))

order = ['Asymptomatic', 'Moderate', 'Severe']
cm = confusion_matrix(y, y_pred, labels=order)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=order)
disp.plot(cmap='Blues')
plt.title('Random Forest (StratifiedGroupKFold) — AHI Class from clinical features', fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix_cv_clinical.png', dpi=300, bbox_inches='tight')
plt.show()