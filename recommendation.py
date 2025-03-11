import faiss
import numpy as np
import pandas as pd
import torch
import requests
from PIL import Image
from io import BytesIO
import os
from typing import List, Dict, Tuple, Optional
from transformers import CLIPProcessor, CLIPModel

class EnhancedRecommendationSystem:
    def __init__(self, model_path='models'):
        self.model_path = model_path
        os.makedirs(model_path, exist_ok=True)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.to(self.device)
        
        self.text_index = None
        self.multimodal_index = None
        
        self.products_df = None
        self.text_embeddings = None
        self.multimodal_embeddings = None
        self._image_products = set()  # Store product IDs with valid images

    def _get_text_embedding(self, text: str) -> np.ndarray:
        inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs.to(self.device))
        return text_features.cpu().numpy().astype('float32')[0]

    def _get_image_embedding(self, image_path: str) -> np.ndarray:
        if image_path.startswith(('http://', 'https://')):
            image = Image.open(BytesIO(requests.get(image_path).content)).convert("RGB")
        else:
            image = Image.open(image_path).convert("RGB")
        
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs.to(self.device))
        return image_features.cpu().numpy().astype('float32')[0]

    def _get_simple_text_embedding(self, product: Dict) -> np.ndarray:
        text = f"{product['name']}, {product['category']}, {product['description']}"
        return self._get_text_embedding(text)

    def _get_multimodal_embedding(self, product: Dict) -> np.ndarray:
        text = f"{product['name']}, {product['category']}"
        text_embedding = self._get_text_embedding(text)
        
        if product.get('image_path'):
            try:
                image_path = product['image_path'].split(',')[0] if ',' in product['image_path'] else product['image_path']
                image_embedding = self._get_image_embedding(image_path)
                combined = (text_embedding + image_embedding) / 2
                self._image_products.add(product['product_id'])  # Track that this product has a valid image
                return combined
            except Exception as e:
                print(f"Error processing image for product {product.get('product_id', 'unknown')}: {str(e)}")
        
        return text_embedding

    def build_index(self, products: List[Dict]):
        self.products_df = pd.DataFrame(products)
        self._image_products = set()  # Reset image product tracking
        
        text_embeddings = []
        multimodal_embeddings = []
        
        for product in products:
            text_embeddings.append(self._get_simple_text_embedding(product))
            
            mm_embedding = self._get_multimodal_embedding(product)
            multimodal_embeddings.append(mm_embedding)
        
        self.text_embeddings = np.array(text_embeddings).astype('float32')
        self.multimodal_embeddings = np.array(multimodal_embeddings).astype('float32')
        
        self.text_embeddings = np.ascontiguousarray(self.text_embeddings)
        self.multimodal_embeddings = np.ascontiguousarray(self.multimodal_embeddings)
        
        faiss.normalize_L2(self.text_embeddings)
        faiss.normalize_L2(self.multimodal_embeddings)
        
        self.text_index = faiss.IndexFlatIP(self.text_embeddings.shape[1])
        self.text_index.add(self.text_embeddings)
        
        self.multimodal_index = faiss.IndexFlatIP(self.multimodal_embeddings.shape[1])
        self.multimodal_index.add(self.multimodal_embeddings)
        
        self._save_data()

    def recommend(self, product_id: str, n: int = 6) -> List[Dict]:
        idx = self.products_df.index[self.products_df['product_id'] == product_id].tolist()
        if not idx:
            raise ValueError(f"Product with ID {product_id} not found")
        
        query = self.text_embeddings[idx[0]].reshape(1, -1)
        faiss.normalize_L2(query)
        distances, indices = self.text_index.search(query, n+1)
        
        return [self.products_df.iloc[i].to_dict() for i in indices[0][1:]]

    def recommend_by_text(self, query_text: str, n: int = 6) -> List[Dict]:
        text_embedding = self._get_text_embedding(query_text)
        
        query = np.ascontiguousarray(text_embedding.reshape(1, -1))
        faiss.normalize_L2(query)
        
        distances, indices = self.text_index.search(query, n)
        
        return [self.products_df.iloc[i].to_dict() for i in indices[0]]

    def recommend_by_image(self, image_path: str, n: int = 6) -> List[Dict]:
        try:
            image_embedding = self._get_image_embedding(image_path)
            
            dummy_text = ""  
            text_embedding = self._get_text_embedding(dummy_text)
            
            query = (image_embedding * 0.8 + text_embedding * 0.2)
            
            query = np.ascontiguousarray(query.reshape(1, -1))
            faiss.normalize_L2(query)
            
            distances, indices = self.multimodal_index.search(query, n)
            
            results = []
            for i in indices[0]:
                results.append(self.products_df.iloc[i].to_dict())
            
            # Prioritize results with images but without adding 'has_image' field
            results_with_images = [r for r in results if r['product_id'] in self._image_products]
            results_without_images = [r for r in results if r['product_id'] not in self._image_products]
            
            return results_with_images + results_without_images[:n-len(results_with_images)]
            
        except Exception as e:
            raise ValueError(f"Image processing failed: {str(e)}")

    def _save_data(self):
        if self.text_index:
            faiss.write_index(self.text_index, os.path.join(self.model_path, 'text_index.faiss'))
        if self.multimodal_index:
            faiss.write_index(self.multimodal_index, os.path.join(self.model_path, 'multimodal_index.faiss'))
        if self.products_df is not None:
            # Save product data without the 'has_image' column
            self.products_df.to_csv(os.path.join(self.model_path, 'products.csv'), index=False)
            
        # Save image products set
        image_products_path = os.path.join(self.model_path, 'image_products.txt')
        with open(image_products_path, 'w') as f:
            for product_id in self._image_products:
                f.write(f"{product_id}\n")

    def load_data(self):
        try:
            self.text_index = faiss.read_index(os.path.join(self.model_path, 'text_index.faiss'))
            self.multimodal_index = faiss.read_index(os.path.join(self.model_path, 'multimodal_index.faiss'))
            self.products_df = pd.read_csv(os.path.join(self.model_path, 'products.csv'))
            
            # Load image products set
            image_products_path = os.path.join(self.model_path, 'image_products.txt')
            self._image_products = set()
            if os.path.exists(image_products_path):
                with open(image_products_path, 'r') as f:
                    for line in f:
                        self._image_products.add(line.strip())
            
            return True
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False