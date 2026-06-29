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

    key = input("\nEnter a key to view its value: ").strip()

    if key == "method":
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
    else:
        probes_times_prof = data["time_profiles"]
        probes_vars = data["var_values"]
        probes_var_changes = data["var_changes"]
        
        if key in probes_times_prof:
            value = probes_times_prof[key]
        elif key in probes_vars:
            value = probes_vars[key]
        elif key in probes_var_changes:
            value = probes_var_changes[key]
        else:
            return

        if type(value) == list:
            while True:
                index = input("\nEnter the index: ").strip()
                if index == "exit":
                    break
                index = int(index)
                if int(index) >= len(value):
                    print("Invalid index! Please enter a valid index")
                    continue
                
                print(f"\nValue of '{key}' at '{index}':\n{value[index]}\n")

if __name__ == "__main__":
    main()
