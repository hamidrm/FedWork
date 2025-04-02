import pickle
import os

def load_pickle_file(filename):
    """Load a dictionary from a pickle file."""
    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def main():
    print("=== Pickle File Reader ===")
    running_path = os.getcwd()  # Get the current working directory
    relative_path = input("Enter the relative path to the pickle file: ").strip()
    filename = os.path.join(running_path, relative_path)  # Concatenate paths

    data = load_pickle_file(filename)
    if data is None:
        print("Failed to load data. Exiting.")
        return

    # Check if "method_info" key exists
    if "method_info" not in data:
        print("Error: 'method_info' key not found in the file.")
        return

    method_info = data["method_info"]
    
    print("\nAvailable keys in 'method_info':")
    for key in method_info.keys():
        print(f" - {key}")

    while True:
        key = input("\nEnter a key to view its value (or type 'exit' to quit): ").strip()
        
        if key.lower() == 'exit':
            break
        elif key in method_info:
            print(f"\nValue of '{key}':\n{method_info[key]}\n")
        else:
            print("Invalid key! Please enter a valid key from the list.")

if __name__ == "__main__":
    main()
