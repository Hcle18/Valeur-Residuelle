import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

from sqlalchemy import distinct
from typing import Dict, List, Optional, Any
import logging

from src.app.data import db, CarData
from src.app.config import FIELD_LABELS, NUMERIC_FIELDS_DROPDOWN

# Initialise dropdown options from the CarData
class DropdownManager:
    def __init__(self):
        self._dropdown_options: Dict[str, List[Dict[str, str]]] = {}
        self._numeric_ranges: Dict[str, Dict[str, float]] = {}
        self._cache_valid = False
        self.logger = logging.getLogger(__name__)

        self.field_labels = FIELD_LABELS
        self.numeric_fields = NUMERIC_FIELDS_DROPDOWN

        # Initialisation automatique
        #self.refresh_all_options()

    @property
    def dropdown_options(self) -> Dict[str, List[Dict[str, str]]]:
        """Accès en lecture seule aux options de dropdown"""
        if not self._cache_valid:
            self.refresh_all_options()
        return self._dropdown_options.copy()
    
    @property
    def numeric_ranges(self) -> Dict[str, Dict[str, float]]:
        """Accès en lecture seule aux plages numériques"""
        if not self._cache_valid:
            self.refresh_all_options()
        return self._numeric_ranges.copy()
    
    def _get_unique_values(self, field_name: str) -> List[Dict[str, str]]:
        """Récupère les valeurs uniques pour un champ donné"""
        try:
            with db.session() as session:
                # Utilisation de getattr pour accéder dynamiquement au champ
                field_attr = getattr(CarData, field_name)
                
                unique_values = session.query(distinct(field_attr)).filter(
                    field_attr.isnot(None)
                ).all()
                
                # Conversion en format dropdown avec tri
                options = [
                    {"label": str(value[0]).strip(), "value": str(value[0]).strip()} 
                    for value in unique_values 
                    if value[0] is not None and str(value[0]).strip()
                ]
                
                return sorted(options, key=lambda x: x['label'])
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des valeurs pour {field_name}: {e}")
                
    def _get_numeric_range(self, field_name: str) -> Dict[str, float]:
        """Récupère les plages min/max pour un champ numérique"""
        try:
            with db.session() as session:
                field_attr = getattr(CarData, field_name)
                
                min_val = session.query(db.func.min(field_attr)).scalar()
                max_val = session.query(db.func.max(field_attr)).scalar()
                
                return {
                    "min": float(min_val) if min_val is not None else 0,
                    "max": float(max_val) if max_val is not None else 100
                }
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération de la plage pour {field_name}: {e}")
     
    def refresh_all_options(self) -> None:
        """Rafraîchit toutes les options depuis la base de données"""
        self.logger.info("Rafraîchissement des options de dropdown...")

        for field_name in self.field_labels.keys():
            self._dropdown_options[field_name] = self._get_unique_values(field_name)
        
        for field_name in self.numeric_fields:
            self._numeric_ranges[field_name] = self._get_numeric_range(field_name)

        self._cache_valid = True
        self.logger.info("Options rafraîchies avec succès")

    def get_options(self, field_name: str) -> List[Dict[str, str]]:
        """Récupère les options pour un champ spécifique"""
        return self.dropdown_options.get(field_name, [])
    
    def get_filtered_modeles(self, selected_marque: str) -> List[Dict[str, str]]:
        """Récupère les modèles filtrés par marque"""
        if not selected_marque:
            return []
        
        try:
            with db.session() as session:
                modeles = session.query(distinct(CarData.modele)).filter(
                    CarData.marque == selected_marque,
                    CarData.modele.isnot(None)
                ).all()
                
                options = [
                    {"label": modele[0].strip(), "value": modele[0].strip()} 
                    for modele in modeles 
                    if modele[0] is not None and modele[0].strip()
                ]
                
                return sorted(options, key=lambda x: x['label'])
                
        except Exception as e:
            self.logger.error(f"Erreur lors du filtrage des modèles pour {selected_marque}: {e}")
            return []
    
    def get_range(self, field_name: str) -> Dict[str, float]:
        """Récupère la plage min/max pour un champ numérique"""
        return self.numeric_ranges.get(field_name, {"min": 0, "max": 100})
    
    def create_dropdown(self, field_name: str, component_id: str = None, 
                       placeholder: str = None, searchable: bool = True, 
                       clearable: bool = True, disabled: bool = False,
                       md_width: int = 4, sm_width: int = 12) -> dbc.Col:
        """Crée un composant dropdown standardisé"""
        
        label = self.field_labels.get(field_name, field_name.title())
        options = self.get_options(field_name)
        
        dropdown_id = component_id or f"dropdown-{field_name}"
        dropdown_placeholder = placeholder or f"Sélectionnez {label.lower()}"
        
        return dbc.Col([
            dbc.Label(label),
            dcc.Dropdown(
                id=dropdown_id,
                options=options,
                value=options[0]['value'] if options else None,
                placeholder=dropdown_placeholder,
                searchable=searchable,
                clearable=clearable,
                disabled=disabled
            )
        ], md=md_width, sm=sm_width)
    
    def create_numeric_input(self, field_name: str, component_id: str = None,
                           input_type: str = "number", placeholder: str = None,
                           md_width: int = 4, sm_width: int = 12) -> dbc.Col:
        """Crée un composant input numérique standardisé"""
        
        label = self.field_labels.get(field_name, field_name.title())
        ranges = self.get_range(field_name)
        
        input_id = component_id or f"input-{field_name}"
        
        return dbc.Col([
            dbc.Label(label),
            dbc.Input(
                id=input_id,
                type=input_type,
                placeholder=placeholder,
                value=ranges.get("min", 0),  # Valeur par défaut
                min=ranges.get("min"),
                max=ranges.get("max")
            )
        ], md=md_width, sm=sm_width)
    
    def invalidate_cache(self) -> None:
        """Invalide le cache pour forcer un rafraîchissement"""
        self._cache_valid = False

dropdown_manager = DropdownManager()