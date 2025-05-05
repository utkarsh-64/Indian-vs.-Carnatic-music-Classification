import streamlit as st
import joblib
import librosa
import numpy as np

# Load the saved model and label encoder (no scaler needed)
model = joblib.load('svm_model.pkl')
label_encoder = joblib.load('label_encoder.joblib')

def home_page():
    st.title("Hindustani vs Carnatic Audio Classification")
    st.write("""
        Welcome to the Hindustani vs Carnatic Audio Classification app. 
        This tool uses machine learning to classify an audio file as either Hindustani or Carnatic music.
    """)

    # Display images for Confusion Matrix, Classification Report, and Spectrograms
    st.header("Model Evaluation Metrics")
    st.image("image.jpg", caption="Confusion Matrix", use_container_width=True)
    st.image("class.png", caption="Classification Report", use_container_width=True)
    st.image("download.jpg", caption="Spectrogram Example", use_container_width=True)

    st.write("""
        1. Mel Spectrogram
What it shows: This graph displays the intensity of audio frequencies over time using the Mel scale, which is perceptually more aligned with how humans hear sound.
Interpretation: The vertical axis shows frequency in the Mel scale (where lower frequencies are represented at the bottom). Bright areas indicate regions with more energy at those frequencies. In this case, the bright bands represent the active harmonic content and rhythmic patterns of the audio. You can see fluctuations indicating changes in the musical dynamics over time.
2. Chroma Spectrogram
What it shows: The chroma spectrogram maps the 12 pitch classes (C, C#, D, etc.) over time, ignoring octave differences. It’s useful for analyzing the harmonic content.
Interpretation: The horizontal bands indicate the prevalence of certain pitch classes in the music over time. This type of plot is helpful for identifying key changes, chords, or harmonic structure. If there are repeated patterns or consistency in certain rows, it indicates the dominance of those pitches, suggesting a strong tonal center.
3. MFCC (Mel-Frequency Cepstral Coefficients) Spectrogram
What it shows: MFCCs are used to capture the timbral aspects of the audio. They represent the short-term power spectrum of the sound, commonly used in audio classification tasks.
Interpretation: The variation in color across the MFCC spectrogram indicates changes in timbre. It helps in distinguishing different instruments or vocal techniques. In this context, you can see that there are gradual changes, which might correspond to variations in the vocalist’s timbre or the introduction of different instruments.
4. Spectral Centroid
What it shows: This graph plots the spectral centroid (the "center of mass" of the spectrum) over time, which is a measure of where the "brightness" of a sound is concentrated.
Interpretation: Higher centroid values (peaks) indicate a higher concentration of high frequencies, suggesting bright, sharp sounds. In contrast, lower values (troughs) correspond to deeper, bass-heavy sounds. Fluctuations in the spectral centroid can indicate changes in instrumentals or vocal techniques.
5. Spectral Contrast
What it shows: The spectral contrast measures the difference in amplitude between peaks and valleys in a spectrum, which reflects the harmonic richness of the audio.
Interpretation: Higher contrast indicates that the audio has strong harmonic content and clear distinctions between frequencies. Lower contrast suggests more noise or uniform sound energy across frequencies. This plot is particularly useful for distinguishing between harmonic music and noise-like sounds.
6. Zero-Crossing Rate
What it shows: This graph measures the rate at which the audio signal changes from positive to negative or vice versa. It is often used as a measure of the noisiness of the signal.
Interpretation: A high zero-crossing rate usually indicates the presence of noise or high-frequency content, while a lower rate indicates smoother, more tonal sounds. In the plot, you can see spikes corresponding to sharp transients or percussive events in the music.
7. Log Spectrogram
What it shows: This is similar to the Mel spectrogram but uses a logarithmic scale for frequencies. It emphasizes lower frequencies, making it easier to detect low-pitched patterns and instruments.
Interpretation: The lower part of the plot is expanded to show more detail in the bass frequencies. It’s helpful when analyzing instruments or vocal ranges that are concentrated in the lower spectrum, providing more clarity than a linear frequency plot.
    """)


# Feature extraction function (same as in your code)
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    features = {
        "chroma_stft_mean": librosa.feature.chroma_stft(y=y, sr=sr).mean(),
        "chroma_stft_std": librosa.feature.chroma_stft(y=y, sr=sr).std(),
        "rmse_mean": librosa.feature.rms(y=y).mean(),
        "rmse_std": librosa.feature.rms(y=y).std(),
        "spectral_centroid_mean": librosa.feature.spectral_centroid(y=y, sr=sr).mean(),
        "spectral_centroid_std": librosa.feature.spectral_centroid(y=y, sr=sr).std(),
        "zero_crossing_rate_mean": librosa.feature.zero_crossing_rate(y).mean(),
        "zero_crossing_rate_std": librosa.feature.zero_crossing_rate(y).std(),
    }

    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    for i in [1, 2, 3, 4, 7, 19, 20]:
        if i in [1, 2, 4, 7, 20]:
            features[f"mfcc_{i}_mean"] = mfccs[i - 1].mean()
        else:
            features[f"mfcc_{i}_std"] = mfccs[i - 1].std()
    return list(features.values())

def model_page():
    st.title("Audio Classification - Hindustani vs Carnatic")
    st.write("Upload an audio file to classify it as either Hindustani or Carnatic.")

    # Upload audio file
    uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav"])

    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Extract features and prepare for prediction
        features = extract_features("temp_audio.wav")

        # Predict using the model (no scaling required)
        prediction = model.predict([features])
        predicted_label = label_encoder.inverse_transform([int(round(prediction[0]))])[0]

        # Display result
        st.write(f"### Predicted Genre: {predicted_label}")

if __name__ == "__main__":
    page = st.sidebar.radio("Go to", ["Home", "Model"])

    if page == "Home":
        home_page()
    else:
        model_page()
