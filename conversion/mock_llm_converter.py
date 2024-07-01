import safetensors
import mediapipe as mp
from safetensors import safe_open, safe_save

class ConversionConfig:
    def __init__(self, input_format, output_format, input_path, output_path):
        self.input_format = input_format
        self.output_format = output_format
        self.input_path = input_path
        self.output_path = output_path

def convert_checkpoint(config: ConversionConfig):
    if config.input_format == 'mediapipe' and config.output_format == 'safetensors':
        # Load the MediaPipe model
        mp_model = mp.solutions.Model(config.input_path)
        weights = mp_model.get_weights()
        
        # Convert weights to a format compatible with safetensors
        tensor_data = {k: v.numpy() for k, v in weights.items()}
        
        # Save the tensor data to a safetensors file
        safe_save(tensor_data, config.output_path)

# Example usage:
config = ConversionConfig(input_format='mediapipe', output_format='safetensors',
                          input_path='path/to/mediapipe/model', output_path='path/to/safetensors/file')
convert_checkpoint(config)
