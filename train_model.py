import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

basedata_noshow = pd.read_csv("medical_appointments_no_shows.csv", dtype = {"PatientId": "str"})

#df_model_data = basedata_noshow.drop(["PatientId", "AppointmentID", "ScheduledDay", "AppointmentDay", "Neighbourhood"],
#                              axis=1)

df_model_data = basedata_noshow[["Gender", "Age", "No-show"]].copy()

# encode target values
label_encode_y = LabelEncoder()
label_encode_y.fit(["No", "Yes"])
df_model_data["No-show"] = label_encode_y.transform(df_model_data["No-show"])


# encode gender
label_encode_gender = LabelEncoder()
label_encode_gender.fit(list(set(df_model_data["Gender"])))
df_model_data["Gender"] = label_encode_gender.transform(list(df_model_data["Gender"]))


# Splitting train test data
y_data = df_model_data["No-show"]
X_data = df_model_data.drop("No-show", inplace=False, axis=1)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.4, random_state=0)

# training randomforest
clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
clf.fit(X_train, y_train)


with open("model_info.pickle", "wb") as f:
    pickle.dump((label_encode_y, label_encode_gender, clf), f)
