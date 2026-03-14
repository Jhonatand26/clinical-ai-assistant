"""
web_scraper.py

Extrae datos tabulares desde fuentes web públicas del dominio
clínico y los persiste como CSV en data/processed/.
"""

import logging
import os
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# URL pública con tabla de medicamentos esenciales OMS
SOURCE_URL = "https://en.wikipedia.org/wiki/" "WHO_Model_List_of_Essential_Medicines"


def fetch_html(url: str) -> BeautifulSoup:
    """
    Descarga el HTML de una URL y retorna el objeto
    BeautifulSoup para parseo.

    Args:
        url: URL de la fuente web.

    Returns:
        Objeto BeautifulSoup con el contenido parseado.

    Raises:
        requests.HTTPError: Si la respuesta no es 200.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    logger.info(f"Fetching URL: {url}")
    response = requests.get(url, timeout=10, headers=headers)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def extract_tables(soup: BeautifulSoup) -> list[pd.DataFrame]:
    """
    Extrae todas las tablas HTML del contenido parseado.

    Args:
        soup: Objeto BeautifulSoup con el HTML.

    Returns:
        Lista de DataFrames, uno por tabla encontrada.
    """
    from io import StringIO

    tables = pd.read_html(StringIO(str(soup)))
    logger.info(f"Found {len(tables)} tables")
    return tables


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia un DataFrame: elimina duplicados, normaliza
    nombres de columnas y elimina filas vacías.

    Args:
        df: DataFrame crudo.

    Returns:
        DataFrame limpio.
    """
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df.drop_duplicates()
    df = df.dropna(how="all")
    return df


def save_dataframe(df: pd.DataFrame, filename: str) -> Path:
    """
    Persiste un DataFrame como CSV en data/processed/.

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
    Orquesta el pipeline completo de extracción web.

    Returns:
        Path del archivo CSV procesado.
    """
    soup = fetch_html(SOURCE_URL)
    tables = extract_tables(soup)

    # Tomamos la tabla más grande — la más relevante
    main_table = max(tables, key=lambda t: t.shape[0])
    clean = clean_dataframe(main_table)

    return save_dataframe(clean, "web_medicines.csv")


if __name__ == "__main__":
    path = run()
    print(f"Extraction complete: {path}")
