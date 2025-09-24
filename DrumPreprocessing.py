import os
import shutil
from pathlib import Path

def filter_rock_files(source_folder, destination_folder, case_sensitive=False):
    """
    Filter files containing the word 'rock' in their filename and copy them to a new dataset folder.
    
    Args:
        source_folder (str): Path to the original folder
        destination_folder (str): Path where filtered files will be copied
        case_sensitive (bool): Whether to perform case-sensitive search (default: False)
    
    Returns:
        dict: Summary of the filtering operation
    """
    
    # Convert paths to Path objects 
    source_path = Path(source_folder)
    dest_path = Path(destination_folder)
    
    # Create destination folder if it doesn't exist yet
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # Statistics tracking
    stats = {
        'total_files_processed': 0,
        'rock_files': 0,
        'files_copied': 0,
        'errors': []
    }
        
    def filename_contains_rock(file_path):
        """Check if filename contains the word 'rock'"""
        filename = file_path.name
        if case_sensitive:
            return 'rock' in filename
        else:
            return 'rock' in filename.lower()
    
    # Walk through all files in the source directory
    for root, dirs, files in os.walk(source_path):
        for file in files:
            file_path = Path(root) / file
            stats['total_files_processed'] += 1
            
            # Check if filename contains 'rock'
            if filename_contains_rock(file_path):
                stats['rock_files'] += 1
                
                try:
                    dest_file_path = dest_path / file_path.name
                    
                    # Handle duplicate filenames by adding a counter
                    counter = 1
                    original_dest = dest_file_path
                    while dest_file_path.exists():
                        stem = original_dest.stem
                        suffix = original_dest.suffix
                        dest_file_path = dest_path / f"{stem}_{counter}{suffix}"
                        counter += 1

                    # Copy the file
                    shutil.copy2(file_path, dest_file_path)
                    stats['files_copied'] += 1
                    
                    print(f"Copied: {file_path.relative_to(source_path)} -> {dest_file_path.name}")
                    
                except Exception as e:
                    stats['errors'].append(f"Error copying {file_path}: {str(e)}")
    
    return stats

def main():
    # Example usage
    source_folder = input("Enter the source folder path: ").strip()
    destination_folder = input("Enter the destination folder path: ").strip()
    
    # Validate source folder exists
    if not os.path.exists(source_folder):
        print(f"Error: Source folder '{source_folder}' does not exist!")
        return
    
    print(f"\nSearching for files with 'rock' in filename in: {source_folder}")
    print(f"Filtered files will be copied to: {destination_folder}")
    print("Processing...\n")
    
    # Run the filtering
    results = filter_rock_files(source_folder, destination_folder)
    
    # Display results
    print("\n" + "="*50)
    print("FILTERING COMPLETE")
    print("="*50)
    print(f"Total files processed: {results['total_files_processed']}")
    print(f"Files with 'rock' in filename: {results['rock_files']}")
    print(f"Files successfully copied: {results['files_copied']}")
    
    if results['errors']:
        print(f"\nErrors encountered: {len(results['errors'])}")
        for error in results['errors']:
            print(f"  - {error}")

if __name__ == "__main__":
    main()