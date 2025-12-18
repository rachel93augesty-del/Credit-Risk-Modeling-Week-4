from pydantic import BaseModel, Field


class CreditRiskInput(BaseModel):
    Amount: float = Field(..., gt=0, description="Transaction amount")
    Value: float = Field(..., gt=0, description="Transaction value")
    CurrencyCode: str = Field(..., min_length=1, description="Currency code, e.g., USD")
    CountryCode: str = Field(..., min_length=2, description="Country code, e.g., US")
    ProviderId: str = Field(..., description="Provider ID")
    ProductCategory: str = Field(..., description="Category of the product")
    ChannelId: str = Field(..., description="Sales channel, e.g., Online")


class CreditRiskOutput(BaseModel):
    risk_label: int = Field(..., description="Predicted label: 0 = Low Risk, 1 = High Risk")
    risk_probability: float = Field(..., ge=0, le=1, description="Predicted probability of high risk")
