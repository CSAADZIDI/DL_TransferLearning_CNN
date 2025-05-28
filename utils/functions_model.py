import os
from tensorflow.keras.models import load_model


def detect_model_architecture(model):
    model_type = type(model).__name__
    model_name = model.name.lower()

    # Direct class type match
    if model_type in ["VGG16", "DenseNet121", "ResNet50V2"]:
        return model_type

    # Heuristic checks for Functional or Sequential models
    layer_names = [layer.name.lower() for layer in model.layers]
        
    if any("block1_conv1" in name for name in layer_names) and any("block5_pool" in name for name in layer_names):
        return "VGG"

    # Check DenseNet patterns
    elif any("conv1/relu" in name or "dense_block" in name or "concat" in name for name in layer_names):
        return "DenseNet"

    # Check ResNet pattern
    elif any("conv2_block1_out" in name or "identity" in name for name in layer_names):
        return "ResNet"

    else:
        return "Ensemble"



def load_models(models_path):
    """ load models

    Args:
        models_path (str): models folder
    Returns:
        model_list (list): models lists
        model_names (list): models names
    """
    model_list = []
    model_names = []
    for file_name in os.listdir(models_path):
        if file_name.endswith(".keras"):  # Only load .keras files
            file_path = os.path.join(models_path, file_name)
            try:
                model = load_model(file_path)
                model_name = os.path.splitext(os.path.basename(file_path))[0]
                model_parts = model_name.split('_')         # ['my', 'model', 'ensemble', 'cnn', 'lstm']
                model_name = ' '.join(model_parts[1:])
                if "dual" in model_name:
                    print("dual")
                print(model_name)
                model_list.append(model)
                model_names.append(model_name)
                print(file_path)
            except Exception as e:
                print(f"Failed to load model from {file_path}: {e}")
    
            

            
    return model_list,model_names
            
    
if __name__ == "__main__":
    """ test func"""
    path= "../models"
    print(load_models(path)[1])