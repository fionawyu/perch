import jax
import jax.numpy as jnp
from perch import PerchModel  # Ensure this import is correct based on your structure

# Load the pretrained model (adjust path if necessary)
model = PerchModel.load_pretrained('path_to_model')  # Replace with actual model path

# Set the model to evaluation mode
model.eval()

# Function to extract embeddings from audio data
def get_embeddings(audio_data):
    with jax.disable_jit():  # Disable JIT for embedding extraction if needed
        embeddings = model(audio_data)  # Pass your audio data here
    return embeddings