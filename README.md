# Dynamic AI: Fractal Universe Chocolate Wafer Model (FUCWM) (V2) (I removed the superweights) 

EDIT: Added think mode. The AI has now a feedback loop for its thoughts in one tab. 

## Video of LM - Studio teaching it to talk

[Video](https://www.youtube.com/live/LK7I3kOZ9AM)

## Overview

Dynamic AI is an experimental neural network model inspired by fractal structures in the universe and the human brain. It incorporates recursive nodes (FractalNodes) to dynamically grow and learn through attention mechanisms and pruning. The model also integrates a VAE (Variational Autoencoder) for encoding latent space representations.

## Key Features

- **Recursive Fractal Nodes**: Nodes grow and create child nodes based on complexity, simulating the recursive, fractal-like nature of neural networks.
- **Variational Autoencoder (VAE)**: Encodes latent representations of inputs.
- **Attention Mechanism**: Dynamically adjusts the focus of the model by assigning importance to different child nodes in the fractal structure.
- **Layer Normalization & Xavier Initialization**: Enhances training stability.
- **Dynamic Complexity-based Growth**: Nodes grow based on complexity thresholds and manage child connections.
- **LM Studio Integration**: Collaborative conversational framework with a local LM Studio instance.
- **Gradio Interface**: User-friendly interface for model interaction, training, and conversation simulation.

## Technical Details

The FUCWM model utilizes a fractal-inspired architecture where each node can spawn child nodes based on input complexity. The attention mechanism allows the model to prioritize certain nodes during the forward pass, enabling more efficient learning and processing. A co-activation matrix tracks how frequently different nodes activate together, further refining the attention scores.

## Requirements

- Python 3.8+
- PyTorch
- Gradio
- OpenAI Python library (for LM Studio integration)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/anttiluode/DaFucV2.git
   cd DaFucV2
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. If using LM Studio, ensure it's installed and running locally. Configure the `lm_studio_client` in the code with your API key and URL.

## Usage

Run the application:
```bash
python app.py
```

This will launch a Gradio interface with the following features:

### 1. Chat with Dynamic AI
- Enter your message and adjust the temperature for response generation.

### 2. Train the Model on Q&A Pairs
- Upload a JSON file with question-answer pairs.
- Set the number of training epochs.
- Monitor training progress and loss metrics.

### 3. LM Studio Conversation
- Set an initial message to start the conversation.
- Define the conversation duration and delay between messages.
- Observe the interaction between the FUCWM model and LM Studio.
- You may have to change the LM Studio model to match the one you use in the code. With old version it works without that.

This line:  {"role": "system", "content": "You're talking to an experimental AI that is still learning to communicate. If it doesn't respond or sends empty messages, please be patient and continue the conversation."}, affects the dialogue a lot. Earlier I had a line "You are talking with experimental fractal model." And the dialogue was steered towards discussions about fractals without a fail.

Change it if you want that not to happen 

### 4. Save/Load Model State
- Save the current model state to a file.
- Load a previously saved state to restore the model.

## Model Configuration

The model's depth and complexity can be adjusted in the main execution:

```python
dynamic_ai = DynamicAI(vocab_size=50000, embed_dim=256, latent_dim=256, output_dim=256, max_depth=7)
```

Note: Increasing the depth can lead to higher complexity and potential instability. Careful tuning is required for optimal performance.

## Training Data Format

Q&A pairs for training should be in a JSON file with the following format:

```json
[
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "Who wrote '1984'?", "answer": "George Orwell"}
]
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
