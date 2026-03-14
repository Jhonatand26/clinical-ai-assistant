"""
legacy_loader.py

Simula la extracción desde un sistema legado clínico
(base de datos SQLite) y persiste los datos limpios
en data/processed/.
"""

import logging
import sqlite3
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
DB_PATH = RAW_DIR / "legacy_clinic.db"


def create_legacy_database() -> None:
    """
    Crea y pobla una base de datos SQLite simulando
    un sistema legado clínico con datos ficticios.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            age INTEGER,
            diagnosis TEXT,
            medication TEXT,
            last_visit TEXT
        )
    """
    )

    sample_data = [
        (1, "Ana García", 45, "Hipertensión", "Losartán 50mg", "2025-01-15"),
        (2, "Carlos Ruiz", 60, "Diabetes tipo 2", "Metformina 850mg", "2025-02-03"),
        (3, "María López", 32, "Asma bronquial", "Salbutamol inhalador", "2025-02-20"),
        (
            4,
            "Jorge Martínez",
            55,
            "Arritmia ventricular",
            "Amiodarona 200mg",
            "2025-03-01",
        ),
        (
            5,
            "Laura Pérez",
            28,
            "Ansiedad generalizada",
            "Sertralina 50mg",
            "2025-03-10",
        ),
    ]

    cursor.executemany(
        "INSERT OR IGNORE INTO patients VALUES (?,?,?,?,?,?)", sample_data
    )

    conn.commit()
    conn.close()
    logger.info(f"Legacy database created at {DB_PATH}")


def extract_from_database() -> pd.DataFrame:
    """
    Extrae todos los registros de pacientes desde
    la base de datos SQLite legada.

    Returns:
        DataFrame con los registros extraídos.

    Raises:
        FileNotFoundError: Si la base de datos no existe.
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Legacy database not found at {DB_PATH}. "
            "Run create_legacy_database() first."
        )

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM patients", conn)
    conn.close()

    logger.info(f"Extracted {len(df)} records from legacy DB")
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el DataFrame: normaliza columnas, elimina
    duplicados y filas con datos críticos faltantes.

    Args:
        df: DataFrame crudo desde la base legada.

    Returns:
        DataFrame limpio.
    """
    df.columns = df.columns.str.strip().str.lower()
    df = df.drop_duplicates(subset=["id"])
    df = df.dropna(subset=["name", "diagnosis"])
    return df


def save_dataframe(df: pd.DataFrame, filename: str) -> Path:
    """
    Persiste el DataFrame limpio como CSV en
    data/processed/.

    Args:
        df: DataFrame a guardar.
        filename: Nombre del archivo CSV.

    Returns:
        Path del archivo guardado.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / filename
    df.to_csv(output_path, index=False)
    logger.info(f"Saved to {output_path}")
    return output_path


def run() -> Path:
    """
    Orquesta el pipeline completo de extracción legada.

    Returns:
        Path del archivo CSV procesado.
    """
    create_legacy_database()
    df = extract_from_database()
    clean = clean_dataframe(df)
    return save_dataframe(clean, "legacy_patients.csv")


if __name__ == "__main__":
    path = run()
    print(f"Extraction complete: {path}")
