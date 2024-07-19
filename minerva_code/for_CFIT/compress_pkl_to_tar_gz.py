import os
import tarfile



def append_extension_to_files(extension=".pkl"):
    current_directory = '/sc/arion/projects/FLAI/marcus/PDAC_Rust_Results/d_ub_100_d_lb_0/Immunogenicity_Dicts/' #os.getcwd()
    print(f"Current directory: {current_directory}")

    for filename in os.listdir(current_directory):
        file_path = os.path.join(current_directory, filename)

        if os.path.isfile(file_path) and not filename.endswith(extension):
            new_filename = f"{filename}{extension}"
            new_file_path = os.path.join(current_directory, new_filename)
            os.rename(file_path, new_file_path)
            print(f"Renamed {filename} to {new_filename}")

def compress_files_in_directory():
    current_directory = '/sc/arion/projects/FLAI/marcus/PDAC_Rust_Results/d_ub_100_d_lb_0/Immunogenicity_Dicts/'
    print(f"Current directory: {current_directory}")

    for filename in os.listdir(current_directory):
        file_path = os.path.join(current_directory, filename)
        print("Checking file: ", file_path)

        if os.path.isfile(file_path) and file_path.endswith(".pkl"):
            base_name, _ = os.path.splitext(filename)
            tar_filename = os.path.join(current_directory, f"{base_name}.tar.gz")

            if os.path.exists(tar_filename):
                print(f"{tar_filename} already exists. Skipping compression.")
                continue

            print(f"Compressing {filename} into {tar_filename}")

            with tarfile.open(tar_filename, "w:gz") as tar:
                tar.add(file_path, arcname=filename)

            print(f"Compressed {filename} into {tar_filename}")
        else:
            print(f"{file_path} is not a file or does not have a .pkl extension.")

if __name__ == "__main__":
    # append_extension_to_files()

    compress_files_in_directory()
