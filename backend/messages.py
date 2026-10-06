"""Text returned with every prediction. The system supports lending staff; it
never refuses an application by itself (proposal sections 3 and 10.4)."""

PREDICTION_LABELS = {True: "Likely to default", False: "Not likely to default"}

SUGGESTED_ACTIONS = {
    "High": "Refer to a senior credit officer; verify income and consider a guarantor or a larger down-payment.",
    "Medium": "Standard review; verify any missing documents.",
    "Low": "Standard processing.",
}

DISCLAIMER = (
    "Academic prototype trained on a historical dataset. Decision support only: "
    "not for automatic refusal. The risk score ranks applicants and is not a probability."
)
