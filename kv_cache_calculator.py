import argparse
import os
import pandas as pd
from huggingface_hub import get_safetensors_metadata, login
from numpy import dtype
from transformers import AutoConfig


def get_model_config(model_name: str):
    """
    Retrieve the configuration of a model from HuggingFace.
    
    Args:
        model_name (str): Name of the model on HuggingFace (e.g., 'bert-base-uncased').
    
    Returns:
        dict: Model configuration as a dictionary, or None if retrieval fails.
    """ 
    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        return config.to_dict()

    except Exception as e:
        print(f"Error retrieving config for {model_name}: {e}")
        return None


def get_model_parameters_count(model_name: str):
    """
    Extract total parameters count using safetensors metadata.

    Args:
        model_name (str): Name of the model on HuggingFace (e.g., 'bert-base-uncased').
    
    Returns:
        int: Total Parameters Count
    """
    try:
        metadata = get_safetensors_metadata(model_name)
        param_count =  metadata.parameter_count
        df = pd.DataFrame.from_dict(param_count, orient='index',columns=['parameters_count'])
        df = df.rename_axis('Precision')
        return df.values.sum()

    except Exception as e:
        print(f"Error retrieving config for {model_name}: {e}")
        return None


def get_kv_cache_parameters(config: dict):
    """
    Extract parameters necessary for KV cache size calculation from the model config.

    Args:
        config (dict): Model configuration dictionary.
    
    Returns:
        tuple: (num_layers, num_heads, head_dim, dtype_size, attention_type, num_kv_heads)
               Returns None if parameters cannot be extracted.
    """
    try:
        # Number of transformer layers
        num_layers = config.get('num_hidden_layers', None)
        
        # Number of query attention heads
        num_heads = config.get('num_attention_heads', None)
        
        # Head dimension (hidden_size / num_heads)
        hidden_size = config.get('hidden_size', None)
        if hidden_size is None or num_heads is None:
            head_dim = None
        else:
            head_dim = hidden_size // num_heads
        
        # Data type size (assuming float32 as default, 4 bytes)
        dtype = config.get('torch_dtype', 'float32')
        dtype_size = 4 if dtype == 'float32' else 2 if dtype == 'float16' else 1 if dtype == 'bfloat16' else 4
        
        # Determine attention type (MHA, MQA, or GQA)
        #MHA (Multi-Head Attention): Each attention head has its own key and value projections, so the KV cache stores keys and values for all heads.
        #MQA (Multi-Query Attention): All heads share a single key and value projection, significantly reducing the KV cache size.
        #GQA (Grouped-Query Attention): Heads are grouped, and each group shares a key and value projection, so the KV cache size depends on the number of groups.

        num_kv_heads = config.get('num_key_value_heads', num_heads)  # Default to num_heads for MHA
        if num_kv_heads == 1:
            attention_type = 'MQA'
        elif num_kv_heads == num_heads:
            attention_type = 'MHA'
        else:
            attention_type = 'GQA'
        
        if None in (num_layers, num_heads, head_dim):
            print("Missing required config parameters (num_hidden_layers, num_attention_heads, or hidden_size).")
            return None
        
        return num_layers, num_heads, head_dim, dtype_size, attention_type, num_kv_heads
    
    except Exception as e:
        print(f"Error extracting KV cache parameters: {e}")
        return None


def calculate_kv_cache_size(
    num_layers: int,
    num_heads: int,
    head_dim: int,
    dtype_size: int,
    attention_type: str,
    num_kv_heads: int,
    batch_size: int,
    sequence_length: int
):
    """
    Calculate the KV cache size for a transformer model based on attention type.
    
    Args:
        num_layers (int): Number of transformer layers.
        num_heads (int): Number of query attention heads.
        head_dim (int): Dimension of each attention head.
        dtype_size (int): Size of the data type in bytes (e.g., 4 for float32).
        attention_type (str): Type of attention mechanism ('MHA', 'MQA', or 'GQA').
        num_kv_heads (int): Number of key/value heads (for GQA/MQA).
        batch_size (int): Batch size.
        sequence_length (int): Sequence length.
    
    Returns:
        int: KV cache size in bytes.
    """
    # KV cache stores key and value for each token, layer, and batch
    # For MHA: 2 (K and V) * num_layers * num_heads * batch_size * sequence_length * head_dim * dtype_size
    # For MQA: 2 (K and V) * num_layers * 1 (single K/V head) * batch_size * sequence_length * head_dim * dtype_size
    # For GQA: 2 (K and V) * num_layers * num_kv_heads * batch_size * sequence_length * head_dim * dtype_size
    if attention_type not in ['MHA', 'MQA', 'GQA']:
        raise ValueError(f"Unsupported attention type: {attention_type}")
    
    # Number of heads used for KV cache
    effective_kv_heads = 1 if attention_type == 'MQA' else num_kv_heads
    
    kv_cache_size = 2 * num_layers * effective_kv_heads * batch_size * sequence_length * head_dim * dtype_size
    return kv_cache_size

def main():
    try:
        # Check if token exists in default cache location
        default_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "token")
        if os.path.exists(default_cache_dir):
            with open(default_cache_dir, "r") as token_file:
                token = token_file.read().strip()
        else:
            raise FileNotFoundError("No token found in default cache location (~/.cache/huggingface/token). "
                                  "Please login using huggingface-cli login or provide a token.")

        # Log into Hugging Face
        login(token=token)
        print("Successfully logged into HuggingFace!")
        
    except Exception as e:
        print(f'Failed to login using cached credentials: {e}')
        print('Please login using huggingface-cli login or provide a token.')
        exit(1)

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Calculate KV cache size for a HuggingFace model.")
    parser.add_argument('--model_name', type=str, required=True, help="HuggingFace model name (e.g., 'bert-base-uncased')")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size")
    parser.add_argument('--sequence_length', type=int, default=2048, help="Sequence length")
    args = parser.parse_args()
    
    # Get model config
    config = get_model_config(args.model_name)
    if config is None:
        print("Failed to retrieve model config. Exiting.")
        return
    
    # Extract KV cache parameters
    kv_params = get_kv_cache_parameters(config)
    if kv_params is None:
        print("Failed to extract KV cache parameters. Exiting.")
        return
    
    num_layers, num_heads, head_dim, dtype_size, attention_type, num_kv_heads = kv_params

    #Get Total Parameters Count
    parameters_count = get_model_parameters_count(args.model_name)
    if 1000000 < parameters_count < 1000000000:
        print(f"Total Parameters for the model:", parameters_count/1000000, "M")
    elif parameters_count > 1000000000:
        print(f"Total Parameters for the model:", parameters_count/1000000000, "B")
    else:
        print(f"Total Parameters for the model:", parameters_count)
    
    # Calculate KV cache size
    kv_cache_size = calculate_kv_cache_size(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        dtype_size=dtype_size,
        attention_type=attention_type,
        num_kv_heads=num_kv_heads,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length
    )
    
    # Convert to human-readable format
    kv_cache_size_gb = kv_cache_size / (1024 ** 3)
    
    # Print results
    print(f"\nModel: {args.model_name}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Sequence Length: {args.sequence_length}")
    print(f"Attention Type: {attention_type}")
    print(f"Number of Layers: {num_layers}")
    print(f"Number of Query Attention Heads: {num_heads}")
    print(f"Number of Key/Value Heads: {num_kv_heads}")
    print(f"Head Dimension: {head_dim}")
    print(f"Data Type Size: {dtype_size} bytes")
    print(f"KV Cache Size: {kv_cache_size:,} bytes ({kv_cache_size_gb:.2f} GB)")


if __name__ == "__main__":
    main()
