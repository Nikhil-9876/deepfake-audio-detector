"""API authentication."""

from fastapi import Header, HTTPException

from config.settings import API_KEY


async def verify_api_key(x_api_key: str = Header(...)):
    """
    Validate API key from request headers.
    
    Args:
        x_api_key: API key from request header
    
    Returns:
        The validated API key
    
    Raises:
        HTTPException: If API key is invalid or missing
    """
    if not x_api_key or len(x_api_key.strip()) == 0:
        raise HTTPException(
            status_code=401,
            detail={
                "status": "error",
                "message": "API key is required in x-api-key header"
            }
        )
    
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail={"status": "error", "message": "Invalid API key"}
        )
    
    return x_api_key
