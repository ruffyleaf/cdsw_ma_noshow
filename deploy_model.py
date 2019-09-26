"""
gender value needs to be either M or F
age value needs to be an integer

"""
import numpy as np
import pickle

# read pickle file
with open("model_info.pickle", "rb") as f:
    label_encode_y, label_encode_gender, clf_model  = pickle.load(f)


# predict
def predict(args):
    gender = args.get("Gender")
    gender = np.int64(label_encode_gender.transform([gender]).item())

    age = np.int64(args.get("Age"))
    pred_y = clf_model.predict([[gender, age]])

    predicted_string = label_encode_y.inverse_transform(pred_y).item()
    return predicted_string
