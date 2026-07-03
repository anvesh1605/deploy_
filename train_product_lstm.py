from __future__ import annotations

from pathlib import Path

import train_release_lstm as base


PRODUCT_INTENTS = [
    "cli_syntax",
    "cli_purpose",
    "cli_parameters",
    "cli_examples",
    "configuration_steps",
    "event_id_meaning",
    "event_id_action",
    "concept_explanation",
    "feature_limitations",
    "product_troubleshooting",
    "show_command_usage",
    "rest_api_usage",
    "snmp_behavior",
    "data_not_available",
    "out_of_domain",
]

PRODUCT_NEGATIVE_ROWS = [
    {
        "input_text": "what is my name?",
        "intent": "out_of_domain",
        "slots": {},
        "target_value": "This is a domain-specific Aruba product documentation assistant, so I cannot answer this question because it is not related to Aruba product documentation.",
        "reference": "This is a domain-specific Aruba product documentation assistant, so I cannot answer this question because it is not related to Aruba product documentation.",
    },
    {
        "input_text": "tell me a joke",
        "intent": "out_of_domain",
        "slots": {},
        "target_value": "This is a domain-specific Aruba product documentation assistant, so I cannot answer this question because it is not related to Aruba product documentation.",
        "reference": "This is a domain-specific Aruba product documentation assistant, so I cannot answer this question because it is not related to Aruba product documentation.",
    },
    {
        "input_text": "what is the weather today?",
        "intent": "out_of_domain",
        "slots": {},
        "target_value": "This is a domain-specific Aruba product documentation assistant, so I cannot answer this question because it is not related to Aruba product documentation.",
        "reference": "This is a domain-specific Aruba product documentation assistant, so I cannot answer this question because it is not related to Aruba product documentation.",
    },
    {
        "input_text": "what is 2 plus 2?",
        "intent": "out_of_domain",
        "slots": {},
        "target_value": "This is a domain-specific Aruba product documentation assistant, so I cannot answer this question because it is not related to Aruba product documentation.",
        "reference": "This is a domain-specific Aruba product documentation assistant, so I cannot answer this question because it is not related to Aruba product documentation.",
    },
    {
        "input_text": "For 9999 AOS-CX 10.18, what CLI syntax is documented for SNMP?",
        "intent": "data_not_available",
        "slots": {"switch": "9999", "version": "10_18", "feature": "SNMP"},
        "target_value": "This particular data is not available in the current Aruba product documentation dataset.",
        "reference": "This particular data is not available in the current Aruba product documentation dataset.",
    },
    {
        "input_text": "For 4100i AOS-CX 10.99, what is the REST API usage for a missing feature?",
        "intent": "data_not_available",
        "slots": {"switch": "4100i", "version": "10_99", "feature": "REST"},
        "target_value": "This particular data is not available in the current Aruba product documentation dataset.",
        "reference": "This particular data is not available in the current Aruba product documentation dataset.",
    },
    {
        "input_text": "For 6200 AOS-CX 10.18, what product documentation command explains SNMP?",
        "intent": "data_not_available",
        "slots": {"switch": "6200", "version": "10_18", "feature": "SNMP"},
        "target_value": "This particular data is not available in the current Aruba product documentation dataset.",
        "reference": "This particular data is not available in the current Aruba product documentation dataset.",
    },
]


base.SELECTED_INTENTS = PRODUCT_INTENTS
base.NEGATIVE_SAMPLE_ROWS = PRODUCT_NEGATIVE_ROWS
base.DEFAULT_DATA_DIR = Path(r"C:\Hpe\Train\Data\product_docs_final")
base.DEFAULT_OUTPUT_DIR = Path(r"C:\Hpe\Train\outputs_product_lstm\all_switches")


if __name__ == "__main__":
    base.main()
