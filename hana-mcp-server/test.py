import os
from dotenv import load_dotenv

print("Starte test.py ...")

# .env laden
loaded = load_dotenv()
print("load_dotenv() ->", loaded)

# HANA Cloud connection configuration
HANA_CONFIG = {
    'address': os.environ.get('HANA_HOST'),
    'port': '443',
    'user': os.environ.get('HANA_USER'),
    'password': os.environ.get('HANA_PASSWORD'),
    'encrypt': True,  # Always use encryption for cloud connections
    'sslValidateCertificate': True
}

# MCP Server Configuration
MCP_CONFIG = {
    'MODEL_SCHEMA': os.environ.get('MODEL_SCHEMA', 'MCP_MODELS'),
    'CONTEXT_SCHEMA': os.environ.get('CONTEXT_SCHEMA', 'MCP_CONTEXTS'),
    'PROTOCOL_SCHEMA': os.environ.get('PROTOCOL_SCHEMA', 'MCP_PROTOCOLS'),
    'max_connections': int(os.environ.get('MAX_CONNECTIONS', '10')),
}

print("\n===== HANA_CONFIG =====")
for k, v in HANA_CONFIG.items():
    print(f"{k}: {v}")

print("\n===== MCP_CONFIG =====")
for k, v in MCP_CONFIG.items():
    print(f"{k}: {v}")

input("\nFertig. Drücke ENTER zum Beenden...")