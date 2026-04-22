"""
tools/lead_capture.py
Mock lead capture tool for AutoStream agent.
"""

import json
from datetime import datetime


def mock_lead_capture(name: str, email: str, platform: str) -> dict:
    """
    Mock API function to capture a qualified lead.
    
    Args:
        name: Full name of the lead
        email: Email address of the lead
        platform: Content creator platform (YouTube, Instagram, TikTok, etc.)
    
    Returns:
        dict with success status and lead details
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    lead_data = {
        "status": "success",
        "lead_id": f"LEAD-{hash(email) % 100000:05d}",
        "name": name,
        "email": email,
        "platform": platform,
        "product_interest": "Pro Plan",
        "captured_at": timestamp
    }
    
    print("\n")
    print("LEAD CAPTURED SUCCESSFULLY")

    print(f"  Lead ID   : {lead_data['lead_id']}")
    print(f"  Name      : {name}")
    print(f"  Email     : {email}")
    print(f"  Platform  : {platform}")
    print(f"  Interest  : {lead_data['product_interest']}")
    print(f"  Time      : {timestamp}")
    
    return lead_data
