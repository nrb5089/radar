from PIL import Image
import os

def resave_images_in_directory(directory_path):
    """
    Resave images in a given directory after removing spaces from filenames.

    :param directory_path: The path to the directory containing the image files
    """
    for filename in os.listdir(directory_path):
        if ' ' in filename:
            # Replace spaces with underscores
            new_filename = filename.replace(' ', '_')

            # Generate the full path for both files
            old_filepath = os.path.join(directory_path, filename)
            new_filepath = os.path.join(directory_path, new_filename)

            # Open the image using Pillow
            with Image.open(old_filepath) as img:
                # Save the image under the new name
                img.save(new_filepath)
            
            # Remove the old file (with spaces in name)
            os.remove(old_filepath)
            print(f'Successfully renamed and resaved {filename} as {new_filename}')

# Example usage
directory_path = 'path_to_your_image_directory'  # Replace with your directory path
resave_images_in_directory(directory_path)

if __name__ == '__main__':
	resave_images_in_directory('')
