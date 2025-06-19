#!/usr/bin/env python3
"""
Script simple para extraer correlaciones de los archivos .pkl y generar la curva.
"""

import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import config

def extract_correlation_from_fold(fold_num, base_output_dir="output", band=None, stimulus=None):
    """
    Extrae la correlación promedio para un fold específico.
    
    Parameters
    ----------
    fold_num : int
        Número del fold
    base_output_dir : str
        Directorio base de output
        
    Returns
    -------
    float or None
        Correlación promedio o None si no se encuentra
    """
    # Probar con ambas configuraciones de banda

    correlation_path = os.path.join(
        base_output_dir, 
        f"exp_fold_{fold_num}",
        "mtrf_ridge_torch", 
        "External", 
        "correlations", 
        "tmin-0.2_tmax0.6", 
        band,
        f"{stimulus}.pkl"
    )
    
    if os.path.exists(correlation_path):
        try:
            with open(correlation_path, 'rb') as f:
                data = pickle.load(f)
            
            if 'average_correlation_subjects' in data:
                correlations = data['average_correlation_subjects']
                mean_correlation = np.nanmean(correlations)
                print(f"Fold {fold_num}: correlation = {mean_correlation:.6f} (band: {band})")
                return mean_correlation
                
        except Exception as e:
            print(f"Error reading {correlation_path}: {e}")
    else:   
        print(f"Warning: No correlation file found for fold {fold_num}")
        return None

def analyze_fold_correlations(start_fold=2, end_fold=75, base_output_dir="output", figs_dir="figures"):
    """
    Analiza las correlaciones para todos los folds y genera la curva.
    """
    # Ensure the figures directory exists
    os.makedirs(figs_dir, exist_ok=True)
    
    for band in config.bands:
        for stimulus in config.stimuli:        
            results = []
            print(f"Analizando correlaciones desde fold {start_fold} hasta {end_fold}...")
            print(f"Banda: {band}, Estímulo: {stimulus}")
            
            for fold_num in range(start_fold, end_fold + 1):
                correlation = extract_correlation_from_fold(fold_num, base_output_dir, band=band, stimulus=stimulus)
                if correlation is not None:
                    results.append({'n_folds': fold_num, 'mean_correlation': correlation})
            
            if not results:
                print("No se encontraron datos de correlación válidos.")
                continue
            
            # Convertir a DataFrame
            df = pd.DataFrame(results)
            print(f"\nDatos extraídos: {len(df)} configuraciones de folds")
            
            # Crear la gráfica
            plt.figure(figsize=(10, 6))
            plt.plot(df['n_folds'], df['mean_correlation'], 'o-', linewidth=2, markersize=6)
            plt.xlabel('Número de Folds')
            plt.ylabel('Correlación Promedio')
            plt.title(f'Correlación vs Número de Folds - {band} - {stimulus}')
            plt.grid(True, alpha=0.3)
            
            # Añadir estadísticas básicas
            max_corr = df['mean_correlation'].max()
            max_fold = df.loc[df['mean_correlation'].idxmax(), 'n_folds']
            plt.axhline(max_corr, color='red', linestyle='--', alpha=0.7, 
                        label=f'Máx: {max_corr:.4f} ({max_fold} folds)')
            
            plt.legend()
            plt.tight_layout()
    
            # Guardar plot con nombre específico por banda y stimulus
            plot_filename = f"correlation_vs_folds_curve_{band}_{stimulus}.png"
            plot_path = os.path.join(figs_dir, plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Gráfica guardada en: {plot_path}")
            plt.close()  # Close the figure to free memory
    
            # Mostrar estadísticas
            print(f"\nEstadísticas:")
            print(f"  Banda: {band}, Estímulo: {stimulus}")
            print(f"  Número de folds analizados: {len(df)}")
            
            print(f"  Correlación máxima: {max_corr:.6f} con {max_fold} folds")
            print(f"  Correlación mínima: {df['mean_correlation'].min():.6f}")
            print(f"  Promedio general: {df['mean_correlation'].mean():.6f}")
            
            # Also save the data as CSV for further analysis
            csv_filename = f"fold_correlations_data_{band}_{stimulus}.csv"
            csv_path = os.path.join(figs_dir, csv_filename)
            df.to_csv(csv_path, index=False)
            print(f"Datos guardados en: {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analizar correlaciones vs folds')
    parser.add_argument('--start_fold', type=int, default=2, help='Fold inicial')
    parser.add_argument('--end_fold', type=int, default=75, help='Fold final')
    parser.add_argument('--base_output_dir', type=str, default="output", help='Directorio base de output')
    parser.add_argument('--figs_dir', type=str, default="figures/analysis/fold_determination", help='Directorio base de figuras')
    
    args = parser.parse_args()
    
    analyze_fold_correlations(args.start_fold, args.end_fold, args.base_output_dir, args.figs_dir)
