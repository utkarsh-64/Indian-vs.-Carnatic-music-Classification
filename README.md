# Hindustani vs Carnatic Audio Classification

A Streamlit web application that uses machine learning to classify audio files as either *Hindustani* or *Carnatic* music. The app extracts audio features using Librosa and predicts the genre using a pre-trained SVM model.

---

## Features

- **Audio Upload & Classification:** Upload an audio file (`.mp3` or `.wav`) and get an instant prediction: Hindustani or Carnatic.
- **Feature Extraction:** Uses spectral and timbral features (e.g., MFCCs, chroma, spectral centroid, zero-crossing rate) for classification.
- **Pre-trained Model:** Utilizes a saved SVM model and label encoder for robust predictions.
- **Simple UI:** Navigate between Home and Model pages using the sidebar.

---

## Installation

### Prerequisites

- Python 3.8 or higher
- The following Python packages:
  - `streamlit`
  - `joblib`
  - `librosa`
  - `numpy`

### Setup

1. **Clone the repository**
   
2. **Install dependencies**
   
3. **Ensure model files are present:**
   - `svm_model.pkl` (Trained SVM model)
   - `label_encoder.joblib` (Label encoder for genre labels)

4. **Run the app:**
   ```bash
   streamlit run model.py
   ```

---

## Usage

- **Home Page:** Introduction and overview of the application.
- **Model Page:** Upload your audio file (`.mp3` or `.wav`). The app will extract features and display the predicted genre.

---

## File Structure

- `model.py` - Main Streamlit app script
- `svm_model.pkl` - Pre-trained SVM model (required)
- `label_encoder.joblib` - Label encoder used for decoding predictions (required)

---

## How It Works

- The app loads the pre-trained SVM model and label encoder at startup.
- When an audio file is uploaded, Librosa extracts relevant features:
  - Chroma STFT (mean, std)
  - Root Mean Square Energy (mean, std)
  - Spectral Centroid (mean, std)
  - Zero Crossing Rate (mean, std)
  - Selected MFCC coefficients (mean or std for MFCCs 1, 2, 3, 4, 7, 19, 20)
- The features are passed to the SVM model for prediction.
- The predicted label is decoded and displayed to the user.

---

## Notes

- Make sure the model and label encoder files are in the same directory as `model.py`.
- The app supports `.mp3` and `.wav` files.
- For best results, use clear audio samples of Hindustani or Carnatic music.

---

## Authors

- Utkarsh Lakhani

---

*This project is for educational purpose.*
