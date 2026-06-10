from typing import Literal

from pydantic import BaseModel, Field


class PropertyFeatures(BaseModel):
    appeal_score: float = Field(..., ge=0.0, le=5.0)
    region_price: float = Field(..., ge=0.0)
    room_type: Literal["Private room", "Entire home/apt", "Shared room"]
    minimum_nights: int = Field(..., gt=0)
    number_of_reviews: int = Field(..., ge=0)
    calculated_host_listings_count: int = Field(..., ge=0)
    availability_365: int = Field(..., ge=0, le=365)
    property_id: int = Field(..., ge=0)
    price_tier: int | None = Field(None, ge=0, le=3)
