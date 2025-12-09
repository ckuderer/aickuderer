# hana_mcp_http_server.py

import os
from contextlib import contextmanager
from typing import List, Dict, Any

from hdbcli import dbapi
import pandas as pd

from mcp.server.fastmcp import FastMCP

# MCP Server
mcp = FastMCP("HANA Cloud MCP", json_response=True)

# HANA Config aus .env
HANA_CONFIG = {
    "address": os.environ.get("HANA_HOST"),
    "port": os.environ.get("HANA_PORT", "443"),
    "user": os.environ.get("HANA_USER"),
    "password": os.environ.get("HANA_PASSWORD"),
    "databaseName": os.environ.get("HANA_DATABASE"),
    "encrypt": True,
    "sslValidateCertificate": False,
}

@contextmanager
def get_hana_connection():
    conn = None
    try:
        conn = dbapi.connect(
            address=HANA_CONFIG["address"],
            port=HANA_CONFIG["port"],
            user=HANA_CONFIG["user"],
            password=HANA_CONFIG["password"],
            databaseName=HANA_CONFIG["databaseName"],
            encrypt=HANA_CONFIG["encrypt"],
            sslValidateCertificate=HANA_CONFIG["sslValidateCertificate"],
        )

        yield conn
    finally:
        if conn is not None:
            conn.close()

# ===== MCP TOOLS =====

@mcp.tool()
def hana_health_check() -> dict:
    """Prüft die Verbindung zur HANA Cloud."""
    try:
        with get_hana_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM DUMMY")
            row = cursor.fetchone()

            cursor.execute("SELECT VERSION FROM M_DATABASE")
            version_row = cursor.fetchone()
            version = version_row[0] if version_row else "unknown"

        return {
            "status": "ok" if row and row[0] == 1 else "error",
            "database_version": version,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }

@mcp.tool()
def hana_run_sql(query: str, max_rows: int = 200) -> List[Dict[str, Any]]:
    """Führt ein SELECT gegen HANA aus."""
    lowered = query.strip().lower()
    if not lowered.startswith("select"):
        raise ValueError("Nur SELECT-Statements sind erlaubt.")

    try:
        with get_hana_connection() as conn:
            sql = f"{query.strip()} LIMIT {int(max_rows)}"
            df = pd.read_sql(sql, conn)
        return df.to_dict(orient="records")
    except Exception as e:
        raise RuntimeError(f"HANA-Query fehlgeschlagen: {e}") from e

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
