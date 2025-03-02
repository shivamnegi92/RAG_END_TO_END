import os

def create_frontend_structure(base_dir="frontend_ui"):
    structure = {
        "components": ["__init__.py", "file_upload.py", "chat_input.py", "response_display.py"],
        "assets": [],
        "config": ["config.py"],
        "services": ["api_service.py"],
        "pages": ["home.py", "about.py"],
        "": ["app.py", "requirements.txt", "README.md"]
    }
    
    for folder, files in structure.items():
        folder_path = os.path.join(base_dir, folder) if folder else base_dir
        os.makedirs(folder_path, exist_ok=True)
        for file in files:
            open(os.path.join(folder_path, file), "w").close()
    
    print(f"Frontend UI folder structure created successfully in '{base_dir}'")

if __name__ == "__main__":
    create_frontend_structure()
