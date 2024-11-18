import nbformat

notebook_path = "C:\junha\GPT_from_Scratch\GPT_from_Scratch.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    notebook = nbformat.read(f, as_version=4)

for cell in notebook.cells:
    if cell.cell_type == "code" and "execution_count" not in cell:
        cell.execution_count = None

with open(notebook_path, "w", encoding="utf-8") as f:
    nbformat.write(notebook, f)

print("Notebook Modified")