from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from Shopforhome.recommendation import Recommendation

class Product(BaseModel):
    product_id: int
    price: float
    category: str
    color: str
    description: str

class RecommendationRequest(BaseModel):
    product_id: int
    n_recommendations: int = 6

app=FastAPI()

recommendation_service = Recommendation()


@app.post("/build-index")
async def build_index(products: List[Product]):
    """Build or rebuild the recommendation index with the provided products"""
    try:
        recommendation_service.build_index([p.dict() for p in products])
        return {"status": "success", "message": "Index built successfully", "product_count": len(products)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/update-index")
async def update_index(products: List[Product]):
    """Update the recommendation index with new products"""
    try:
        recommendation_service.update_index([p.dict() for p in products])
        return {"status": "success", "message": "Index updated successfully", "products_added": len(products)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/recommend")
async def get_recommendations(request: RecommendationRequest):
    """Get product recommendations based on a product ID"""
    try:
        recommendations = recommendation_service.get_recommendations(
            request.product_id, 
            request.n_recommendations
        )
        return {"status": "success", "recommendations": recommendations}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/products", response_model=List[Product])
async def get_products():
    """Get all products in the database"""
    if recommendation_service.products_df is None:
        return []
    products = recommendation_service.products_df.to_dict(orient='records')
    return products