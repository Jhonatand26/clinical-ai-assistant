from abc import ABC, abstractmethod
from typing import List, Dict

class BaseLoader(ABC):
    """
    Clase base abstracta para los cargadores de datos (Loaders).
    Define un contrato común que todos los cargadores deben seguir.
    El método `load` debe implementarse y devolver los datos en un
    formato estandarizado: una lista de diccionarios.
    """

    @abstractmethod
    def load(self) -> List[Dict]:
        """
        Carga los datos desde la fuente (DB, web, API, etc.)
        y los devuelve en un formato estandarizado.

        Returns:
            Una lista de diccionarios, donde cada diccionario
            representa un registro o documento.
        """
        pass
