import faiss
import numpy as np
import pandas as pd
import os
import json
from typing import List, Dict, Union
from sentence_transformers import SentenceTransformer

class Recommendation:
    def __init__(self, model_path='models'):
        
        self.model_path = model_path
        os.makedirs(model_path, exist_ok=True)
        
        self.text_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.index = None
        self.products_df = None
        self.feature_embeddings = None
        self.feature_cols = None
        
        self.load_data()
    
    def preprocess_features(self, products: List[Dict]) -> pd.DataFrame:
        df = pd.DataFrame(products)
        df['price_normalized'] = (df['price'] - df['price'].min()) / (df['price'].max() - df['price'].min())
        df = pd.get_dummies(df, columns=['category', 'color'])

        descriptions = df['description'].tolist()
        desc_embeddings = self.text_model.encode(descriptions)

        for i in range(desc_embeddings.shape[1]):
            df[f'desc_embedding_{i}'] = desc_embeddings[:, i]

        return df
    
    def build_index(self, products: List[Dict]):
        self.products_df = pd.DataFrame(products)
        processed_df = self.preprocess_features(products)

        self.feature_cols = processed_df.select_dtypes(include=[np.number]).columns
        self.feature_embeddings = processed_df[self.feature_cols].values.astype('float32')
        self.feature_embeddings = np.ascontiguousarray(self.feature_embeddings)
        
        faiss.normalize_L2(self.feature_embeddings)

        self.index = faiss.IndexFlatIP(self.feature_embeddings.shape[1])
        self.index.add(self.feature_embeddings)
        
        self.save_data()

    def get_recommendations(self, product_id: int, n_recommendations: int = 6) -> List[Dict]:
        if self.index is None:
            raise ValueError("Index not built. Please call build_index() first")
    
        product_idx = self.products_df.index[self.products_df['product_id'] == product_id].tolist()
        if not product_idx:
            raise ValueError(f"Product with id {product_id} not found")
    
        query_vector = self.feature_embeddings[product_idx[0]].reshape(1, -1)

        distances, indices = self.index.search(query_vector, n_recommendations + 1)

        indices = indices[0][1:]  
        distances = distances[0][1:]

        recommendations = []
        for idx, score in zip(indices, distances):
            product = self.products_df.iloc[idx].to_dict()
            # product['similarity_score'] = float(score)
            recommendations.append(product)

        return recommendations
    
    def update_index(self, new_products: List[Dict]):
        if self.index is None:
            self.build_index(new_products)
        else:
            new_products_df = pd.DataFrame(new_products)
            self.products_df = pd.concat([self.products_df, new_products_df], ignore_index=True)
            
            new_df = self.preprocess_features(new_products)
            new_embeddings = new_df[self.feature_cols].values.astype('float32')
            
            faiss.normalize_L2(new_embeddings)
            self.index.add(new_embeddings)
            
            self.feature_embeddings = np.vstack([self.feature_embeddings, new_embeddings])
            
            self.save_data()
    
    def save_data(self):
        
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(self.model_path, 'product_index.faiss'))
        
        if self.products_df is not None:
            self.products_df.to_csv(os.path.join(self.model_path, 'products.csv'), index=False)
        
        if self.feature_embeddings is not None:
            np.save(os.path.join(self.model_path, 'feature_embeddings.npy'), self.feature_embeddings)
            
        if self.feature_cols is not None:
            with open(os.path.join(self.model_path, 'feature_cols.json'), 'w') as f:
                json.dump(list(self.feature_cols), f)
    
    def load_data(self):
        index_path = os.path.join(self.model_path, 'product_index.faiss')
        products_path = os.path.join(self.model_path, 'products.csv')
        embeddings_path = os.path.join(self.model_path, 'feature_embeddings.npy')
        feature_cols_path = os.path.join(self.model_path, 'feature_cols.json')
        
        if all(os.path.exists(p) for p in [index_path, products_path, embeddings_path, feature_cols_path]):
            try:
                self.index = faiss.read_index(index_path)
                self.products_df = pd.read_csv(products_path)
                self.feature_embeddings = np.load(embeddings_path)
                
                with open(feature_cols_path, 'r') as f:
                    self.feature_cols = json.load(f)
                
                print("Successfully loaded index and data")
                return True
            except Exception as e:
                print(f"Error loading data: {e}")
        
        return False