# datei: test_hana_connect.py
import os
from dotenv import load_dotenv
from hdbcli import dbapi

load_dotenv()

conn = dbapi.connect(
    address=os.environ.get('HANA_HOST'),
    port=443,
    user=os.environ.get('HANA_USER'),
    password=os.environ.get('HANA_PASSWORD'),
    encrypt=True,
    sslValidateCertificate=True
)

cursor = conn.cursor()
cursor.execute("SELECT CURRENT_USER, CURRENT_SCHEMA FROM DUMMY")
print(cursor.fetchone())
conn.close()
