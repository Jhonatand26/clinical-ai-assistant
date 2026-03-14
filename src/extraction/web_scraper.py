"""
web_scraper.py

Extrae datos tabulares desde fuentes web públicas del dominio
clínico y los persiste como CSV en data/processed/.
Refactorizado para implementar BaseLoader.
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict
from io import StringIO # Importar StringIO para pd.read_html

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.extraction.base import BaseLoader # Importar BaseLoader

PROCESSED_DIR = Path("data/processed") # Se mantiene por si hay otras funciones de procesamiento/guardado aquí

# URL pública con tabla de medicamentos esenciales OMS (original del usuario)
SOURCE_URL = "https://en.wikipedia.org/wiki/" "WHO_Model_List_of_Essential_Medicines"


class WebScraper(BaseLoader):
    """
    Implementación de BaseLoader que extrae la lista de medicamentos
    esenciales de la OMS a través de web scraping de Wikipedia.
    """
    def __init__(self, url: str = SOURCE_URL):
        self.url = url
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }

    def _fetch_html(self) -> BeautifulSoup:
        """
        Descarga el HTML de la URL configurada y retorna el objeto
        BeautifulSoup para parseo.
        """
        logger.info(f"Fetching URL: {self.url}")
        response = requests.get(self.url, timeout=10, headers=self.headers)
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")

    def _extract_tables(self, soup: BeautifulSoup) -> list[pd.DataFrame]:
        """
        Extrae todas las tablas HTML del contenido parseado.
        """
        tables = pd.read_html(StringIO(str(soup)))
        logger.info(f"Found {len(tables)} tables")
        return tables

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia un DataFrame: elimina duplicados, normaliza
        nombres de columnas y elimina filas vacías.
        """
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        df = df.drop_duplicates()
        df = df.dropna(how="all")
        return df

    def load(self) -> List[Dict]:
        """
        Orquesta el pipeline completo de extracción web y devuelve
        los datos como una lista de diccionarios.
        """
        try:
            soup = self._fetch_html()
            tables = self._extract_tables(soup)

            if not tables:
                logger.warning("No tables found on the page.")
                return []

            # Tomamos la tabla más grande — la más relevante
            main_table = max(tables, key=lambda t: t.shape[0])
            cleaned_df = self._clean_dataframe(main_table)

            medicines_list = cleaned_df.to_dict(orient="records")
            logger.info(f"Se extrajeron {len(medicines_list)} medicamentos.")
            return medicines_list
        except requests.RequestException as e:
            logger.error(f"Error durante el scraping de {self.url}: {e}")
            return []
        except Exception as e:
            logger.error(f"Un error inesperado ocurrió durante el scraping: {e}")
            return []


if __name__ == "__main__":
    scraper = WebScraper()
    data = scraper.load()
    if data:
        print("\nEjemplo de datos extraídos por WebScraper:")
        print(data[0])
    else:
        print("No se extrajeron datos.")
