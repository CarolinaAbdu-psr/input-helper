import json
import logging
import pandas as pd
from typing import Dict, Any, Tuple

# Configure logging
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)


class PropertyExtractor:
    """Handles extraction of properties from objects into a graph structure."""
    
    def __init__(self, graph):
        self.graph = graph
        self.properties = {}
    
    def extract_dataframe_properties(self, obj, key: str, parent_node: str) -> None:
        """
        Extract properties from dataframe columns and create nodes for time series data.
        
        Args:
            obj: Object containing the dataframe
            key: Property key to extract dataframe from
            parent_node: Parent node ID to connect to
        """
        try:
            logger.debug(f"Starting dataframe extraction for key '{key}' from parent node '{parent_node}'")
            df = obj.get_df(key)
            logger.debug(f"Successfully loaded dataframe for key '{key}' with {len(df.columns)} columns")
            
            for col_name in df.columns:
                try:
                    series = df[col_name]
                    time_series_data = self._create_time_series_dict(series)
                    
                    full_id = self._generate_node_id(obj, col_name)
                    property_type = f"{key}"
                    
                    self._add_property_node(full_id, property_type, obj, parent_node, time_series_data)
                    logger.debug(f"Successfully added property node '{full_id}' for column '{col_name}'")
                except Exception as col_error:
                    logger.error(f"Error processing column '{col_name}' in dataframe for key '{key}': {col_error}", exc_info=True)
                
        except Exception as e:
            logger.error(f"Error extracting dataframe properties for key '{key}': {e}", exc_info=True)
    
    def _create_time_series_dict(self, series: pd.Series) -> Dict[str, Any]:
        """Convert pandas Series to dictionary format."""
        return {str(date): value for date, value in series.items()}
    
    def _generate_node_id(self, obj, col_name: str) -> str:
        """Generate unique node ID from object and column name."""
        return f"{col_name}".replace(" ", "")
    
    def _add_property_node(self, node_id: str, property_type: str, obj, 
                          parent_node: str, time_series_data: Dict) -> None:
        """Add a property node to the graph with metadata."""
        try:
            self.graph.add_node(node_id, type=property_type, object=obj)
            self.graph.add_edge(parent_node, node_id, title="Has_Property")
            logger.debug(f"Added node '{node_id}' of type '{property_type}' with edge to parent '{parent_node}'")
        
            time_series_data.update({
                'Full_ID': node_id,
                'ObjType': property_type,
                'code': obj.code,
                'name': f"{obj.type}_{obj.code}"
            })
            
            self.properties[node_id] = time_series_data
            logger.debug(f"Stored time series data for node '{node_id}' with {len(time_series_data)} entries")
        except Exception as e:
            logger.error(f"Error adding property node '{node_id}': {e}", exc_info=True)
            raise
    
    def extract_static_property(self, key: str, obj) -> Any:
        """
        Extract and sanitize static property values.
        
        Args:
            key: Property key
            obj: Source object
            
        Returns:
            Sanitized property value
        """
       
        value = obj.get(key)
        
        if isinstance(value, str):
            return value.rstrip()
        
        if isinstance(value, (int, float, bool, list, dict, type(None))):
            return value
        
        # Convert complex types to string
        try:
            return str(value).rstrip()
        except:
            return "Complex non-serializable value"
        
    def extract_static_property_with_dimensions(self, key: str, obj) -> Any:

        dimensions = obj.description(key).dimensions()
        for d in dimensions.values():
            value = obj.get(f"{key}({d})")
    
    def extract_dynamic_property(self, key: str, obj, node_id: str) -> Dict[str, Any]:
        """
        Extract dynamic properties from dataframes.
        
        Args:
            key: Property key
            obj: Source object
            node_id: Current node ID
            
        Returns:
            Dictionary of extracted properties
        """
        result = {}
        
        try:
            logger.debug(f"Starting dynamic property extraction for key '{key}' on node '{node_id}'")
            df = obj.get_df(key)
            
            if df.empty:
                logger.debug(f"Dataframe for key '{key}' is empty, setting value to 0")
                result[key] = 0
            elif len(df) == 1:
                logger.debug(f"Dataframe for key '{key}' has single row, extracting values")
                result.update(self._extract_single_row_values(df))
            else:
                logger.debug(f"Dataframe for key '{key}' has {len(df)} rows, extracting multi-row values and creating property nodes")
                result.update(self._extract_multi_row_values(df)) 
                self.extract_dataframe_properties(obj, key, node_id)
            
            logger.debug(f"Successfully extracted dynamic property '{key}' with {len(result)} properties")
                
        except Exception as e:
            logger.error(f"Error processing dynamic property '{key}' on node '{node_id}': {e}", exc_info=True)
        
        return result
    
    def _extract_single_row_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract values from single-row dataframe."""
        return {col_name: df[col_name].iloc[0] for col_name in df.columns}
    
    def _extract_multi_row_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract cnd data"""
        result = {}
        default_date = "01/01/1900"
        
        if default_date in df.index:
            for col_name in df.columns:
                result[col_name] = df[col_name].loc[default_date]
        
        return result
    
    def is_extractable_property(self, obj, key: str) -> bool:
        """Check if property should be extracted based on description."""
        try:
            desc = obj.description(key)
            return not desc.is_reference()
        except:
            return False
    
    def extract_node_properties(self, original_graph):
        """
        Extract all properties from graph nodes.
        
        Args:
            original_graph: Source graph to extract from
            
        Returns:
            Tuple of (modified graph, properties dictionary)
        """
        logger.info(f"Starting node properties extraction for graph with {len(original_graph.nodes())} nodes")
        self.graph = original_graph.copy(as_view=False)
        self.properties = {}
        
        error_count = 0
        for node_id, attrs in original_graph.nodes(data=True):
            try:
                obj = attrs["object"]
                logger.debug(f"Processing node '{node_id}' of type '{attrs.get('type', 'unknown')}'")
                
                # Initialize base properties
                node_props = self._create_base_properties(node_id, attrs, obj)
                
                # Extract additional properties
                try:
                    descriptions = obj.descriptions()
                    logger.debug(f"Found {len(descriptions)} descriptions for node '{node_id}'")
                    filtered_props = self._extract_all_properties(descriptions, obj, node_id)
                    node_props.update(filtered_props)
                    logger.debug(f"Extracted {len(filtered_props)} properties for node '{node_id}'")
                except Exception as e:
                    error_count += 1
                    logger.error(f"Error inspecting attributes of node '{node_id}': {e}", exc_info=True)
                
                self.properties[node_id] = node_props
            except Exception as e:
                error_count += 1
                logger.error(f"Critical error processing node '{node_id}': {e}", exc_info=True)
        
        logger.info(f"Completed node properties extraction. Total nodes: {len(original_graph.nodes())}, Errors: {error_count}")
        return self.graph, self.properties
    
    def _create_base_properties(self, node_id: str, attrs: Dict, obj) -> Dict[str, Any]:
        """Create base property dictionary for a node."""
        return {
            'Full_ID': node_id,
            'ObjType': attrs['type'],
            'code': obj.code,
            'name': obj.name.strip() 
        }
    
    def _extract_all_properties(self, descriptions: Dict, obj, node_id: str) -> Dict[str, Any]:
        """Extract all valid properties from object descriptions."""
        filtered_data = {}
        skipped_count = 0
        static_count = 0
        dynamic_count = 0
        
        for key in descriptions.keys():
            try:
                if not self.is_extractable_property(obj, key):
                    skipped_count += 1
                    logger.debug(f"Skipping non-extractable property '{key}' on node '{node_id}'")
                    continue
                
                desc = obj.description(key)
                
                if not desc.is_dynamic() and len(obj.description(key).dimensions()) == 0 :
                    # Static property
                    logger.debug(f"Extracting static property without dimensions '{key}' from node '{node_id}'")
                    filtered_data[key] = self.extract_static_property(key, obj)
                    static_count += 1
                
                else:
                    # Dynamic property
                    logger.debug(f"Extracting dynamic property '{key}' from node '{node_id}'")
                    dynamic_props = self.extract_dynamic_property(key, obj, node_id)
                    filtered_data.update(dynamic_props)
                    dynamic_count += 1
            except Exception as e:
                logger.warning(f"Error extracting property '{key}' from node '{node_id}': {e}", exc_info=True)
        
        logger.debug(f"Property extraction summary for node '{node_id}': {static_count} static, {dynamic_count} dynamic, {skipped_count} skipped")
        return filtered_data


def extract_node_properties(graph_original):
    """
    Main entry point for extracting node properties.
    
    Args:
        graph_original: NetworkX graph with object nodes
        
    Returns:
        Tuple of (modified graph, properties dictionary)
    """
    extractor = PropertyExtractor(graph_original)
    return extractor.extract_node_properties(graph_original)


if __name__== "__main__":
    from data_loader import data_loader
    logger.setLevel(logging.DEBUG)
    STUDY_PATH = r"D:\\01-Repositories\\factory-graphs\\Bolivia"
    G, load_times = data_loader(STUDY_PATH)
    G, node_properties = extract_node_properties(G)