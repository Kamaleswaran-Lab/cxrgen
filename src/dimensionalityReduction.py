import cudf
import cupy as cp
import numpy as np
import pandas as pd
from cuml.manifold import UMAP, TSNE
from cuml.decomposition import PCA
from cuml.preprocessing import StandardScaler, MinMaxScaler, normalize
from sklearn.model_selection import ParameterGrid

import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional, Any
import time
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import os
import pickle

class RapidsDimReduction:
    def __init__(self, data: cudf.DataFrame):
        """
        Initialize the dimensionality reduction framework.
        
        Args:
            data: Input DataFrame (should be preprocessed and scaled)
        """
        self.data = data
        self.reduced_data = {}
        self.reduction_times = {}
        self.models = {}
        
    def reduce_umap(self, n_components: int = 2, **kwargs) -> Dict[str, Any]:
        """
        Perform UMAP dimensionality reduction.
        
        Args:
            n_components: Number of components to reduce to
            **kwargs: Additional parameters for UMAP
        """
        start_time = time.time()
        
        default_params = {
            'n_neighbors': 15,
            'min_dist': 0.1,
            'random_state': 42
        }
        params = {**default_params, **kwargs}
        
        umap = UMAP(n_components=n_components, output_type = 'numpy',  **params)
        reduced_data = umap.fit_transform(self.data)
        
        # Convert to pandas for visualization
        reduced_df = pd.DataFrame(
            reduced_data,
            columns=[f'UMAP_{i+1}' for i in range(n_components)]
        )
        
        time_taken = time.time() - start_time
        
        self.reduced_data['UMAP'] = reduced_df
        self.reduction_times['UMAP'] = time_taken
        self.models['UMAP'] = umap
        
        return {
            'reduced_data': reduced_df,
            'time': time_taken,
            'model': umap
        }
    
    def reduce_tsne(self, n_components: int = 2, **kwargs) -> Dict[str, Any]:
        """
        Perform t-SNE dimensionality reduction.
        
        Args:
            n_components: Number of components to reduce to
            **kwargs: Additional parameters for t-SNE
        """
        start_time = time.time()
        
        default_params = {
            'perplexity': 30,
            'random_state': 42
        }
        params = {**default_params, **kwargs}
        
        tsne = TSNE(n_components=n_components, output_type = 'numpy',  **params)
        reduced_data = tsne.fit_transform(self.data)
        
        reduced_df = pd.DataFrame(
            reduced_data,
            columns=[f'TSNE_{i+1}' for i in range(n_components)]
        )
        
        time_taken = time.time() - start_time
        
        self.reduced_data['TSNE'] = reduced_df
        self.reduction_times['TSNE'] = time_taken
        self.models['TSNE'] = tsne
        
        return {
            'reduced_data': reduced_df,
            'time': time_taken,
            'model': tsne
        }
    
    def reduce_pca(self, n_components: int = 2, **kwargs) -> Dict[str, Any]:
        """
        Perform PCA dimensionality reduction.
        
        Args:
            n_components: Number of components to reduce to
            **kwargs: Additional parameters for PCA
        """
        start_time = time.time()
        
        default_params = {
            'random_state': 42
        }
        params = {**default_params, **kwargs}
        
        pca = PCA(n_components=n_components, output_type = 'numpy',  **params)
        reduced_data = pca.fit_transform(self.data)
        
        reduced_df = pd.DataFrame(
            reduced_data,
            columns=[f'PC_{i+1}' for i in range(n_components)]
        )
        
        explained_variance = pca.explained_variance_ratio_
        
        time_taken = time.time() - start_time
        
        self.reduced_data['PCA'] = reduced_df
        self.reduction_times['PCA'] = time_taken
        self.models['PCA'] = pca
        
        return {
            'reduced_data': reduced_df,
            'time': time_taken,
            'model': pca,
            'explained_variance': explained_variance
        }
    
    def plot_2d(self, 
                method: str, 
                labels: Optional[np.ndarray] = None, 
                title: Optional[str] = None) -> go.Figure:
        """
        Create an interactive 2D scatter plot of the reduced data.
        
        Args:
            method: Reduction method ('UMAP', 'TSNE', or 'PCA')
            labels: Cluster labels (optional)
            title: Plot title (optional)
        """
        if method not in self.reduced_data:
            raise ValueError(f"Method {method} not found in reduced data")
            
        data = self.reduced_data[method]
        
        if labels is not None:
            data['Cluster'] = labels
            
        fig = px.scatter(
            data,
            x=data.columns[0],
            y=data.columns[1],
            color='Cluster' if labels is not None else None,
            title=title or f"{method} Visualization",
            template='plotly_white'
        )
        
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(
            width=800,
            height=600,
            legend_title_text='Cluster' if labels is not None else None
        )
        
        return fig
    
    def plot_3d(self, 
                method: str, 
                labels: Optional[np.ndarray] = None, 
                title: Optional[str] = None) -> go.Figure:
        """
        Create an interactive 3D scatter plot of the reduced data.
        
        Args:
            method: Reduction method ('UMAP', 'TSNE', or 'PCA')
            labels: Cluster labels (optional)
            title: Plot title (optional)
        """
        if method not in self.reduced_data:
            raise ValueError(f"Method {method} not found in reduced data")
            
        data = self.reduced_data[method]
        
        if data.shape[1] < 3:
            raise ValueError(f"Need 3 components for 3D plot, but got {data.shape[1]}")
            
        if labels is not None:
            data['Cluster'] = labels
            
        fig = px.scatter_3d(
            data,
            x=data.columns[0],
            y=data.columns[1],
            z=data.columns[2],
            color='Cluster' if labels is not None else None,
            title=title or f"{method} 3D Visualization",
            template='plotly_white'
        )
        
        fig.update_traces(marker=dict(size=3))
        fig.update_layout(
            width=900,
            height=700,
            legend_title_text='Cluster' if labels is not None else None
        )
        
        return fig
    
    def plot_pca_explained_variance(self) -> go.Figure:
        """
        Create a plot of explained variance ratio for PCA.
        """
        if 'PCA' not in self.models:
            raise ValueError("PCA has not been performed yet")
            
        explained_variance = self.models['PCA'].explained_variance_ratio_.get()
        cumulative_variance = np.cumsum(explained_variance)
        
        fig = go.Figure()
        
        # Bar plot for individual explained variance
        fig.add_trace(go.Bar(
            x=list(range(1, len(explained_variance) + 1)),
            y=explained_variance,
            name='Individual'
        ))
        
        # Line plot for cumulative explained variance
        fig.add_trace(go.Scatter(
            x=list(range(1, len(cumulative_variance) + 1)),
            y=cumulative_variance,
            name='Cumulative',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title='PCA Explained Variance Ratio',
            xaxis_title='Principal Component',
            yaxis_title='Explained Variance Ratio',
            template='plotly_white',
            width=800,
            height=500
        )
        
        return fig
    
def main():
    root = Path('/hpc/group/kamaleswaranlab/EmoryDataset/Images/chest_xrays')
    embedding_path = root / 'BioMedCLIP_embeddings'
    save_path = root / 'dimReduce'
    os.makedirs(save_path, exist_ok=True)

    # Load the embeddings
    files = os.listdir(embedding_path)
    embeddings = [cp.load(embedding_path / file).flatten() for file in files]
    embedding_matrix = cp.vstack(embeddings)
    print("Embedding matrix shape:", embedding_matrix.shape)
    np.save(embedding_path / 'embedding_matrix.npy', embedding_matrix)

    # Scale the embeddings to norm 1 for each row 
    embedding_matrix = normalize(embedding_matrix, norm='l2', axis=1)

    reducer = RapidsDimReduction(embedding_matrix)

    # Perform UMAP reduction
    param_range = {
            'n_neighbors': [20, 25, 30,  35, 40, 45, 50],
            'min_dist': [0.1, 0.3, 0.5],
            'metric': ['manhattan', 'euclidean']
        }

    param_grid = ParameterGrid(param_range)

    for param in param_grid:
        print("UMAP parameters:", param)
        result = reducer.reduce_umap(n_components=2, **param)
        print("Time taken:", result['time'])
        
        # Save the reduced data
        save_file = save_path / f"umap_{param['n_neighbors']}_{param['min_dist']}_{param['metric']}.csv"
        result['reduced_data'].to_csv(save_file, index=False)
        print("Saved to:", save_file)

        # Save the model 
        model_file = save_path / f"umap_{param['n_neighbors']}_{param['min_dist']}_{param['metric']}_model.pkl"

        with open(model_file, 'wb') as f:
            pickle.dump(result['model'], f)

        print("Model saved to:", model_file)

    # Perform t-SNE reduction
    param_range = {
        'perplexity': [30, 40, 50, 60, 70, 80, 90],
        'learning_rate': [10, 50, 100, 200, 500]
    }

    param_grid = ParameterGrid(param_range)

    for param in param_grid:
        print("t-SNE parameters:", param)
        result = reducer.reduce_tsne(n_components=2, **param)
        print("Time taken:", result['time'])
        
        # Save the reduced data
        save_file = save_path / f"tsne_{param['perplexity']}_{param['learning_rate']}.csv"
        result['reduced_data'].to_csv(save_file, index=False)
        print("Saved to:", save_file)

        # Save the model 
        model_file = save_path / f"tsne_{param['perplexity']}_{param['learning_rate']}_model.pkl"

        with open(model_file, 'wb') as f:
            pickle.dump(result['model'], f)

        print("Model saved to:", model_file)

    # Perform PCA reduction
    result = reducer.reduce_pca(n_components=2)
    print("Time taken:", result['time'])

    # Save the reduced data
    save_file = save_path / "pca.csv"
    result['reduced_data'].to_csv(save_file, index=False)
    print("Saved to:", save_file)

    # Save the model
    model_file = save_path / "pca_model.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(result['model'], f)

    print("Model saved to:", model_file)                

if __name__ == "__main__":
    main()                                                                                    
        


