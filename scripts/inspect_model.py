import torch

# Path to your .bin file
model_path = "/mnt/extra/models/iot2023/pytorch_model.bin"

# Load the state dictionary
try:
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    print("Successfully loaded the model's state dictionary.")

    # Get the layer names (keys of the dictionary)
    layer_names = state_dict.keys()

    print("\nLayers provided in the .bin file:")
    for name in layer_names:
        print(name)

except FileNotFoundError:
    print(f"Error: The file at '{model_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")