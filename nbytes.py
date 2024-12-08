from ggml_type_traits import type_traits

# Constants
GGML_MAX_DIMS = 4
# ne = {96, 96, 96, 64},
# nb = {10, 1, 1, 1},
# blck_size = 0x20

# current ggml_nbytes= 0x11b

def ggml_blck_size(type_name):
    trait = type_traits[type_name]
    return 32 if isinstance(trait['blck_size'], str) else trait['blck_size']

def ggml_type_size(type_name):
    trait = type_traits[type_name]
    return 32 if isinstance(trait['type_size'], str) else trait['type_size']

def ggml_nbytes(ne, nb, type_name):
    blck_size = ggml_blck_size(type_name)
    if blck_size == 1:
        nbytes = ggml_type_size(type_name)
        for i in range(GGML_MAX_DIMS):
            nbytes += (ne[i] - 1) * nb[i]
    else:
        nbytes = ne[0] * nb[0] // blck_size
        for i in range(1, GGML_MAX_DIMS):
            nbytes += (ne[i] - 1) * nb[i]
    return nbytes

def main():
    print("Enter the tensor's dimensions (ne) and strides (nb) for each of the 4 dimensions:")
    
    ne = []
    nb = []
    
    for i in range(GGML_MAX_DIMS):
        ne_val = int(input(f"Enter ne[{i}] (size for dimension {i}): "))
        nb_val = int(input(f"Enter nb[{i}] (stride for dimension {i}): "))
        ne.append(ne_val)
        nb.append(nb_val)
    
    print("\nAvailable types:", list(type_traits.keys()))
    type_name = input("Enter the tensor type (e.g., GGML_TYPE_F32): ")
    
    if type_name not in type_traits:
        print("Invalid type entered!")
        return
    
    result = ggml_nbytes(ne, nb, type_name)
    print(f"\nThe calculated number of bytes for the tensor is: {result}")

if __name__ == "__main__":
    main()