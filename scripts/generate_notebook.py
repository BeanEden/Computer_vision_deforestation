import nbformat as nbf
import os

def create_forest_expert_notebook():
    nb = nbf.v4.new_notebook()

    # --- STYLE & HEADER ---
    header = """# Écosysteme d'Analyse Forestière Haute Résolution
## Système Expert de Monitoring de la Résilience et de l'Anthropisation
---
Ce notebook orchestre le pipeline modulaire pour l'analyse multi-parcelles (Lot 1 & Lot 2)."""
    
    nb.cells.append(nbf.v4.new_markdown_cell(header))

    # --- IMPORTS ---
    code_import = """import os
import sys
import pandas as pd
# Import du pipeline modulaire
sys.path.append(os.path.abspath('..'))
import src.pipeline as pipe
import matplotlib.pyplot as plt

%matplotlib inline"""
    nb.cells.append(nbf.v4.new_code_cell(code_import))

    # --- EXECUTION LOT 1 ---
    nb.cells.append(nbf.v4.new_markdown_cell("## 1. Analyse du Groupe de Parcelles 1 (Lot 1)"))
    code_lot1 = "df1 = pipe.run_full_forest_analysis('../data', 'Lo1_A', 'Lot1_B', 'lot1')\ndf1.head()"
    nb.cells.append(nbf.v4.new_code_cell(code_lot1))

    # --- STYLE TABLEAU LOT 1 ---
    code_style1 = """# Affichage avec terminologie experte et style
df1_styled = df1.copy()
df1_styled.columns = ['Zone', 'Surface t0', 'Surface t1', 'Variation %', 'Diagnostic']
df1_styled.style.background_gradient(cmap='RdYlGn_r', subset=['Variation %'])\\
        .format({'Variation %': '{:.2f}%'})"""
    nb.cells.append(nbf.v4.new_code_cell(code_style1))

    # --- EXECUTION LOT 2 ---
    nb.cells.append(nbf.v4.new_markdown_cell("## 2. Analyse du Groupe de Parcelles 2 (Lot 2)"))
    code_lot2 = "df2 = pipe.run_full_forest_analysis('../data', 'Lot2_A', 'Lot2_B', 'lot2')\ndf2.head()"
    nb.cells.append(nbf.v4.new_code_cell(code_lot2))

    # --- STYLE TABLEAU LOT 2 ---
    code_style2 = """df2_styled = df2.copy()
df2_styled.columns = ['Zone', 'Surface t0', 'Surface t1', 'Variation %', 'Diagnostic']
df2_styled.style.background_gradient(cmap='RdYlGn_r', subset=['Variation %'])\\
        .format({'Variation %': '{:.2f}%'})"""
    nb.cells.append(nbf.v4.new_code_cell(code_style2))

    # Save
    notebook_path = 'notebooks/analyse_deforestation.ipynb'
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    
    print(f"Notebook expert généré : {notebook_path}")

if __name__ == "__main__":
    create_forest_expert_notebook()
