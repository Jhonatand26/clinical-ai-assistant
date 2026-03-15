"""
legacy_loader.py

Simula la extracción desde un sistema legado clínico
(base de datos SQLite) y proporciona una clase cargador
que implementa BaseLoader.
"""

import sys
import logging
import sqlite3
from pathlib import Path
from typing import List, Dict

# No necesitamos pandas aquí si solo devolvemos List[Dict]
# import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.extraction.base import BaseLoader # Importar BaseLoader

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed") # Se mantiene por si hay otras funciones de procesamiento/guardado aquí

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


class LegacyLoader(BaseLoader):
    """
    Implementación de BaseLoader para extraer datos de pacientes
    desde la base de datos SQLite del sistema legado.
    """
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        if not self.db_path.exists():
            # Asegurarse de que la DB exista antes de intentar cargar
            create_legacy_database()
            if not self.db_path.exists(): # Verificar de nuevo por si create_legacy_database falla
                raise FileNotFoundError(f"La base de datos no se encontró y no pudo ser creada en: {self.db_path}")

    def load(self) -> List[Dict]:
        """
        Se conecta a la base de datos SQLite, extrae los registros de pacientes
        y los devuelve en el formato estándar (lista de diccionarios).
        """
        logger.info(f"Conectando a la base de datos legado en {self.db_path}...")
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row # Para obtener resultados como diccionarios

        cursor = conn.cursor()
        cursor.execute("SELECT id, name, age, diagnosis, medication, last_visit FROM patients")
        patients = cursor.fetchall()
        conn.close()

        patient_list = [dict(row) for row in patients]
        logger.info(f"Se extrajeron {len(patient_list)} registros de pacientes desde la DB legado.")
        return patient_list


if __name__ == "__main__":
    # Asegurarse de que la base de datos exista para la prueba
    # create_legacy_database() # Ya se llama en el __init__ del loader si no existe
    
    loader = LegacyLoader()
    data = loader.load()
    if data:
        print("\nEjemplo de datos extraídos por LegacyLoader:")
        print(data[0])
    else:
        print("No se extrajeron datos.")
