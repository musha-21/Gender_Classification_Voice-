import numpy as np
import librosa
from tensorflow.keras.models import load_model

# SAME VALUES AS TRAINING
N_MFCC = 20
MAX_LEN = 44
SR = 16000

LABELS = {
    0: "male",
    1: "female"
}

MODEL_PATH = "gender_cnn.h5"

print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded.")


def extract_mfcc(file_path, n_mfcc=N_MFCC, max_len=MAX_LEN, sr=SR):
    """
    Ye function TRAINING jaise hi hoga.
    """
    y, sr = librosa.load(file_path, sr=sr)
    # silence remove
    y, _ = librosa.effects.trim(y, top_db=20)
    # normalize
    y = librosa.util.normalize(y)
    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # pad/crop
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0,0),(0,max_len - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]

    # mfcc normalization
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-9)

    return mfcc


def predict_gender_from_file(file_path):
    """
    Ye function ek audio file se gender predict karega.
    """
    mfcc = extract_mfcc(file_path)
    x = mfcc.reshape(1, N_MFCC, MAX_LEN, 1)
    prob = model.predict(x)[0][0]  # sigmoid output

    label_id = int(prob >= 0.5)    # 0 or 1
    label = LABELS[label_id]

    return label, float(prob)


if __name__ == "__main__":
    # Test ke liye:
    test_file = "mic_input.wav"  # yaha koi bhi file naam daal sakte ho
    print("Testing on:", test_file)
    label, prob = predict_gender_from_file(test_file)
    print("Predicted:", label, "| prob =", prob)
