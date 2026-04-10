import os
import cv2
import pandas as pd
import numpy as np
import src.data_loader as dl
import src.visualization as vis
import src.preprocessing as pre
import src.features as ft
import src.clustering as cl
import src.postprocessing as post
import src.quantification as quant
from typing import Dict, Any

def categorize_loss(val: float) -> str:
    """Categorizes the environmental impact based on the loss percentage.
    
    Thresholds:
        - > 20%: Rupture d'Intégrité (Critique)
        - > 0%: Anthropisation Avérée
        - > -15%: Résilience Écosystémique (Stable)
        - else: Succession Écologique (Régénération)

    Args:
        val: Loss percentage (positive means loss).

    Returns:
        A clinical diagnostic label based on forestry standards.
    """
    if val > 20: return "Rupture d'Intégrité (Critique)"
    if val > 0: return 'Anthropisation Avérée'
    if val > -15: return 'Résilience Écosystémique (Stable)'
    return 'Succession Écologique (Régénération)'

def run_expert_zone_analysis(p0: str, p1: str, name: str, lot_id: str = 'lot1', k: int = 3) -> Dict[str, Any]:
    """Executes the full automated image processing pipeline for a single study zone.
    
    Steps: Acquisition, Preprocessing, Feature Extraction, Clustering (IA), 
    Morphology, and Impact Quantification.

    Args:
        p0: Path to the image at t0.
        p1: Path to the image at t1.
        name: Identifier for the study zone.
        lot_id: Identifier for the parcel group (for storage).
        k: Number of clusters for segmentation.

    Returns:
        A dictionary containing the quantified ecological results.
    """
    print(f"\n{'#'*70}")
    print(f"RAPPORT D'ANALYSE EXPERITEK : ZONE {name} ({lot_id.upper()})")
    print(f"{'#'*70}")
    
    lot_dir = os.path.join('outputs', lot_id)
    os.makedirs(lot_dir, exist_ok=True)
    
    # --- S1: Acquisition ---
    img_a, img_b = dl.load_pair((p0, p1))
    
    # --- S2: Prétraitement et Filtrage ---
    img_gauss = pre.denoise_image(img_a, method='gaussian', kernel_size=3)
    psnr_gauss = cv2.PSNR(img_a, img_gauss)
    
    img_a_p = pre.preprocess(img_a)
    img_b_p = pre.preprocess(img_b)
    
    # --- S3: Détection & Segmentation IA ---
    f0 = ft.get_feature_vector(img_a_p)
    f1 = ft.get_feature_vector(img_b_p)
    
    l0, centers_0 = cl.run_kmeans(f0, n_clusters=k)
    l1, centers_1 = cl.run_kmeans(f1, n_clusters=k)
    
    veg_idx_0 = cl.identify_vegetation_cluster(f0, l0, centers_0)
    veg_idx_1 = cl.identify_vegetation_cluster(f1, l1, centers_1)
    m0 = cl.get_vegetation_mask(l0, veg_idx_0, img_a.shape[:2])
    m1 = cl.get_vegetation_mask(l1, veg_idx_1, img_b.shape[:2])
    
    # --- S4: Morphologie & Primitives ---
    m0_c = post.apply_morphology(m0)
    m1_c = post.apply_morphology(m1)
    
    # SAUVEGARDE DES MASQUES (Version Coul. Jaune Canopée)
    mask_rgb = np.zeros((m0_c.shape[0], m0_c.shape[1], 3), dtype=np.uint8)
    mask_rgb[m0_c > 0] = [0, 255, 255] # BGR Yellow
    cv2.imwrite(os.path.join(lot_dir, f"mask0_{name}"), mask_rgb)
    
    mask_rgb_1 = np.zeros((m1_c.shape[0], m1_c.shape[1], 3), dtype=np.uint8)
    mask_rgb_1[m1_c > 0] = [0, 255, 255] # BGR Yellow
    cv2.imwrite(os.path.join(lot_dir, f"mask1_{name}"), mask_rgb_1)
    
    cv2.imwrite(os.path.join(lot_dir, f"orig0_{name}"), cv2.cvtColor(img_a, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(lot_dir, f"orig1_{name}"), cv2.cvtColor(img_b, cv2.COLOR_RGB2BGR))
    
    # --- S5: Interprétation & Impact ---
    res = quant.compute_deforestation_stats(m0_c, m1_c)
    res['image'] = name
    
    c_map = quant.get_change_map(m0_c, m1_c)
    save_path = os.path.join(lot_dir, f"expert_change_{name}")
    vis.plot_change_map(c_map, save_path=save_path)
    
    loss = res['loss_percentage']
    print(f"Analyse terminée. Perte de biomasse : {loss:.2f}%")
    
    return res

def run_full_forest_analysis(data_path: str = 'data', 
                             folder_a: str = 'Lo1_A', 
                             folder_b: str = 'Lot1_B', 
                             lot_id: str = 'lot1') -> pd.DataFrame:
    """Orchestrates the global analysis for an entire parcel group (lot).

    Args:
        data_path: Root data directory.
        folder_a: Folder containing images at t0.
        folder_b: Folder containing images at t1.
        lot_id: Unique identifier for the lot.

    Returns:
        A pandas DataFrame summarizing all results for the lot.
    """
    abs_data_path = os.path.abspath(data_path)
    pairs = dl.get_image_pairs(data_path, folder_a, folder_b)
    
    if not pairs:
        print(f"ERROR: No image pairs found in {abs_data_path}/{folder_a}")
        return pd.DataFrame(columns=['image', 'surface_t0', 'surface_t1', 'loss_percentage', 'Verdict'])
        
    all_stats = []
    for p0, p1 in pairs:
        name = os.path.basename(p0)
        stats = run_expert_zone_analysis(p0, p1, name, lot_id=lot_id)
        all_stats.append(stats)
        
    df = pd.DataFrame(all_stats)
    if not df.empty:
        df['Verdict'] = df['loss_percentage'].apply(categorize_loss)
    
    lot_dir = os.path.join('outputs', lot_id)
    os.makedirs(lot_dir, exist_ok=True)
    csv_path = os.path.join(lot_dir, 'bilan_final_expert.csv')
    df.to_csv(csv_path, index=False)
    
    # Technical Audit Production
    audit_path = os.path.join(lot_dir, 'TECHNICAL_AUDIT.md')
    audit_content = f"""# Audit Technique - Pipeline Forestier v2.0 ({lot_id.upper()})
Fichier généré automatiquement pour validation par l'équipe technique.

## ⚙️ Configuration du Pipeline
- **Prétraitement** : Filtrage Gaussien (3x3) + CLAHE (Luminance Lab).
- **Segmentation** : K-Means multispectral (k=3) avec identification auto du cluster canopée.
- **Morphologie** : Opérations d'Ouverture et Fermeture (Noyau Carré 5x5).
- **Statistiques** : Calcul de variation binaire inter-temporelle.

## 📊 Résumé Qualité
- Nombre de zones traitées : {len(df)}
- Statut : Validation IA terminée.
"""
    with open(audit_path, 'w', encoding='utf-8') as f:
        f.write(audit_content)
    
    return df
