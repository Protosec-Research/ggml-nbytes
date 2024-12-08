from nbytes import ggml_nbytes, type_traits
from colorama import init, Fore, Style

def format_bytes(bytes_value):
    """Format bytes value with color based on size"""
    if bytes_value < 1024:  # < 1KB
        return f"{Fore.GREEN}{bytes_value:,} B{Style.RESET_ALL}"
    elif bytes_value < 1024**2:  # < 1MB
        return f"{Fore.YELLOW}{bytes_value/1024:.2f} KB{Style.RESET_ALL}"
    elif bytes_value < 1024**3:  # < 1GB
        return f"{Fore.RED}{bytes_value/1024**2:.2f} MB{Style.RESET_ALL}"
    else:  # >= 1GB
        return f"{Fore.MAGENTA}{bytes_value/1024**3:.2f} GB{Style.RESET_ALL}"

def test_combinations():
    # Initialize colorama for Windows compatibility
    init()
    
    # Initialize results dictionary
    results = {}

    # Test dimensions
    test_sizes = [1, 32, 64, 96, 128, 256, 512, 1024]
    test_strides = [1, 2, 4, 8, 16]

    print(f"\n{Fore.CYAN}=== GGML Memory Usage Analysis ==={Style.RESET_ALL}\n")

    for type_name, traits in type_traits.items():
        if traits['type_name'] == 'DEPRECATED' or 'REMOVED' in traits['type_name']:
            continue

        results[type_name] = {
            'type_info': traits,
            'configurations': []
        }

        print(f"\n{Fore.BLUE}Type: {type_name}{Style.RESET_ALL}")
        print(f"{'Dimensions':<30} {'Strides':<30} {'Memory':<20}")
        print("-" * 80)

        for size in test_sizes:
            configs = []
            
            # Test 1D tensor
            ne = [size, 1, 1, 1]
            nb = [1, 1, 1, 1]
            bytes_1d = ggml_nbytes(ne, nb, type_name)
            print(f"{str(ne):<30} {str(nb):<30} {format_bytes(bytes_1d)}")
            configs.append({
                'shape': 'tensor_1d',
                'dimensions': ne,
                'strides': nb,
                'bytes': bytes_1d
            })

            # Test 2D square tensor
            ne = [size, size, 1, 1]
            nb = [1, size, 1, 1]
            bytes_2d = ggml_nbytes(ne, nb, type_name)
            print(f"{str(ne):<30} {str(nb):<30} {format_bytes(bytes_2d)}")
            configs.append({
                'shape': 'tensor_2d',
                'dimensions': ne,
                'strides': nb,
                'bytes': bytes_2d
            })

            # Test 3D cube tensor
            ne = [size, size, size, 1]
            nb = [1, size, size*size, 1]
            bytes_3d = ggml_nbytes(ne, nb, type_name)
            print(f"{str(ne):<30} {str(nb):<30} {format_bytes(bytes_3d)}")
            configs.append({
                'shape': 'tensor_3d',
                'dimensions': ne,
                'strides': nb,
                'bytes': bytes_3d
            })

            results[type_name]['configurations'].append({
                'size': size,
                'tensors': configs
            })

            print("-" * 80)
    
    return results

def main():
    import json
    from datetime import datetime
    
    results = test_combinations()
    
    # Save results to JSON file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ggml_memory_analysis_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{Fore.GREEN}Results saved to: {filename}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
