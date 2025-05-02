# 🖱️ Human-like Mouse Movement Generator

This project captures, trains, and simulates human-like mouse movement using LSTM and Transformer-based deep learning models. Originally designed to support UX research and behavioral simulation, it provides a complete pipeline from real user data collection to synthetic pointer movement replay.

## 📦 Project Structure
├── main.py # Collects human mouse movement data via Pygame
├── dataNormalization.py # Normalizes recorded mouse trajectories
├── mouseLSTM.py # LSTM-based model training
├── mouse_transformer_model.py # Transformer model definition
├── transformerTrain.py # Transformer model training script
├── moveMouse.py # Replay with LSTM/Transformer-generated paths
├── mouse_data/ # Raw JSON data from gameplay
├── mouse_data_normalized/ # Normalized training data
├── models/ # Saved model checkpoints

## 🚀 Features

- Record user mouse trajectories while clicking on random circles
- Normalize (dt, dx, dy) data for training
- Train LSTM and Transformer models on mouse movement patterns
- Replay generated paths using PyAutoGUI to simulate natural movement
- Fine control over replay characteristics (steering, jitter, substeps)

## 🧰 Requirements

Install the necessary dependencies:

pip install -r requirements.txt

## 🎮 Usage
1. Record Mouse Movements
python main.py ( Output: mouse_data/mouse_data_<timestamp>.json)
2. Normalize Data
python dataNormalization.py ( Output: mouse_data_normalized/normalized_mouse_data_<timestamp>.json)
3. Train an LSTM Model
python mouseLSTM.py ( Model saved to: models/mouse_lstm_checkpoint.pth)
4. Train a Transformer Model
python transformerTrain.py( Model saved to: models/mouse_transformer.pth)
5. Simulate Mouse Movement
python moveMouse.py (Make sure to update target_pos in moveMouse.py for your screen resolution.

## ⚠️ Disclaimer
This tool is intended for research and automation testing only.

Do not use it to automate clicks in sensitive or unauthorized environments.

Ensure all data collection is done with informed user consent.
## 📜 License
MIT License