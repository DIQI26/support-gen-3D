from .slc_support_structure import slc_main
from .stl_algo import stl_main


import os

def get_file_type(file_path):
    return file_path[-3:].lower()

def get_file():
    # Get the directory of the current Python script
    script_dir = os.path.dirname(os.path.realpath(__file__))
    src_dir = os.path.join(script_dir, "src")  # Adjust this based on the folder structure

    while True:
        file_name = input("Please enter your file name: ")
        file_path = os.path.join(src_dir, file_name)  # Construct the full path to the file
        
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"The file '{file_name}' does not exist in the 'src' folder. Please try again.")
            continue

        file_type = get_file_type(file_name)

        if file_type == "stl":
            stl_main(file_path)
            break
        elif file_type == "slc":
            slc_main(file_path)
            break
        else:
            print("Incorrect file type, please input a valid STL or SLC file.")
            
def main():
    get_file()

if __name__ == "__main__":
    main()
