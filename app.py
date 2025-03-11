from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Query
from pydantic import BaseModel
from typing import List, Optional
from recommendation import EnhancedRecommendationSystem
import tempfile
import requests
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class Product(BaseModel):
    product_id: str
    name: str
    price: float
    category: str
    description: str
    image_path: Optional[str] = None
    stock: int

class RecommendationRequest(BaseModel):
    product_id: str
    n_recommendations: int = 6

class TextSearchRequest(BaseModel):
    query: str
    n_recommendations: int = 6

app = FastAPI(title="Product Recommendation API")
recommendation_service = EnhancedRecommendationSystem()


@app.on_event("startup")
async def startup_event():
    recommendation_service.load_data()

@app.post("/build-index")
async def build_index(products: List[Product]):
    """Build both text and multimodal indices from product data"""
    try:
        response=requests.get("https://shopforhome-production.up.railway.app/api/products/getallproducts")
        response.raise_for_status()
        products=response.json()

        recommendation_service.build_index(products)
        return {
            "status": "success", 
            "product_count": len(products),
            "message": "Built indices successfully"
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.post("/recommend")
async def recommend_products(request: RecommendationRequest):
    """Get product recommendations using text-based features"""
    try:
        results = recommendation_service.recommend(
            request.product_id, 
            request.n_recommendations
        )
        return {
            "recommendations": results,
            # "recommendation_type": "text-based"
        }
    except ValueError as e:
        raise HTTPException(404, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.post("/recommend-by-text")
async def recommend_by_text(request: TextSearchRequest):
    """Get product recommendations using a text query"""
    try:
        results = recommendation_service.recommend_by_text(
            request.query,
            request.n_recommendations
        )
        return {
            "recommendations": results,
            # "recommendation_type": "text-query",
            # "query": request.query
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.post("/recommend-by-image")
async def recommend_by_image(
    image: UploadFile = File(...), 
    n: int = Query(6, description="Number of recommendations to return")
):
    """Get product recommendations using an image query"""
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await image.read())
            results = recommendation_service.recommend_by_image(tmp.name, n)
        return {
            "recommendations": results
            # "recommendation_type": "image-based"
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))
    finally:
        try:
            os.remove(tmp.name)
        except:
            pass

@app.get("/products")
async def get_products():
    """Get all products in the database"""
    if recommendation_service.products_df is None:
        return []
    return recommendation_service.products_df.to_dict('records')
