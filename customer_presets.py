"""Sample customer profiles used to pre-fill the prediction form.

Each preset is a complete, valid customer record (it passes
prediction_pipeline.validate_customer_input) plus a short description
that is shown in the app.
"""

from __future__ import annotations

from typing import Any, Dict

CUSTOMER_PRESETS: Dict[str, Dict[str, Any]] = {
    "High risk: new fiber customer": {
        "description": (
            "Senior customer in their first month on a month-to-month fiber plan, "
            "paying by electronic check with no security or support add-ons."
        ),
        "values": {
            "gender": "Female",
            "SeniorCitizen": 1,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 1,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 95.50,
            "TotalCharges": 95.50,
        },
    },
    "Borderline: month-to-month DSL": {
        "description": (
            "Just over a year with the company on DSL without a long-term contract. "
            "The model puts this customer close to the 50% decision threshold."
        ),
        "values": {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 14,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Mailed check",
            "MonthlyCharges": 64.00,
            "TotalCharges": 896.00,
        },
    },
    "Low risk: loyal family on 2-year contract": {
        "description": (
            "Four years with the company, two-year contract, automatic bank payments "
            "and all protection / support add-ons."
        ),
        "values": {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": 48,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Bank transfer (automatic)",
            "MonthlyCharges": 55.20,
            "TotalCharges": 2650.00,
        },
    },
    "Low risk: phone-only customer": {
        "description": (
            "Long-standing customer with a basic phone plan and no internet service, "
            "paying by credit card on a two-year contract."
        ),
        "values": {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": 60,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "No",
            "OnlineSecurity": "No internet service",
            "OnlineBackup": "No internet service",
            "DeviceProtection": "No internet service",
            "TechSupport": "No internet service",
            "StreamingTV": "No internet service",
            "StreamingMovies": "No internet service",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Credit card (automatic)",
            "MonthlyCharges": 20.05,
            "TotalCharges": 1203.00,
        },
    },
}

# Values the form starts with before any preset is picked
DEFAULT_PRESET = "Low risk: loyal family on 2-year contract"


def get_preset_values(name: str) -> Dict[str, Any]:
    """Returns a copy of a preset's field values (so callers can't mutate the original)."""
    if name not in CUSTOMER_PRESETS:
        raise KeyError(f"Unknown preset '{name}'. Available: {list(CUSTOMER_PRESETS)}")
    return dict(CUSTOMER_PRESETS[name]["values"])
