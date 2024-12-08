from nbytes import ggml_nbytes, ggml_type_size, ggml_blck_size, type_traits
from itertools import product
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

def find_tensor_configurations(target_bytes, max_dim=128):
    """Find possible tensor configurations that result in the target bytes"""
    init()  # Initialize colorama
    
    possible_dims = range(1, max_dim + 1)
    results = {}

    print(f"\n{Fore.CYAN}=== Searching for configurations that match {format_bytes(target_bytes)} ==={Style.RESET_ALL}\n")

    for type_name, traits in type_traits.items():
        if traits['type_name'] == 'DEPRECATED' or 'REMOVED' in traits['type_name']:
            continue

        matches = []
        
        # Test 1D configurations
        for size in possible_dims:
            ne = [size, 1, 1, 1]
            nb = [1, 1, 1, 1]
            bytes_1d = ggml_nbytes(ne, nb, type_name)
            if bytes_1d == target_bytes:
                matches.append({
                    'shape': '1D',
                    'dimensions': ne.copy(),
                    'strides': nb.copy()
                })

        # Test 2D configurations
        for d1, d2 in product(possible_dims, repeat=2):
            ne = [d1, d2, 1, 1]
            nb = [1, d1, 1, 1]
            bytes_2d = ggml_nbytes(ne, nb, type_name)
            if bytes_2d == target_bytes:
                matches.append({
                    'shape': '2D',
                    'dimensions': ne.copy(),
                    'strides': nb.copy()
                })

        # Test 3D configurations
        for d1, d2, d3 in product(range(1, min(max_dim, 32)), repeat=3):
            ne = [d1, d2, d3, 1]
            nb = [1, d1, d1*d2, 1]
            bytes_3d = ggml_nbytes(ne, nb, type_name)
            if bytes_3d == target_bytes:
                matches.append({
                    'shape': '3D',
                    'dimensions': ne.copy(),
                    'strides': nb.copy()
                })

        if matches:
            results[type_name] = matches
            print(f"\n{Fore.BLUE}Type: {type_name}{Style.RESET_ALL}")
            print(f"{'Shape':<8} {'Dimensions':<30} {'Strides':<30}")
            print("-" * 70)
            
            for match in matches:
                print(f"{match['shape']:<8} {str(match['dimensions']):<30} {str(match['strides'])}")

    return results

def main():
    import json
    from datetime import datetime

    try:
        target_bytes = int(input("Enter target size in bytes: "))
        max_dim = int(input("Enter maximum dimension size to search (default 128): ") or "128")
        
        results = find_tensor_configurations(target_bytes, max_dim)
        
        if not results:
            print(f"\n{Fore.RED}No configurations found for {format_bytes(target_bytes)}{Style.RESET_ALL}")
            return

        # Save results to JSON file with timestamp
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # filename = f"reverse_ggml_analysis_{timestamp}.json"
        
        # with open(filename, 'w') as f:
        #     json.dump(results, f, indent=2)
        
        # print(f"\n{Fore.GREEN}Results saved to: {filename}{Style.RESET_ALL}")

    except ValueError as e:
        print(f"{Fore.RED}Error: Please enter valid numbers{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
