import json
import pandas

def extract_df_properties(obj,key,G, properties,node):
    try : 
        df = obj.get_df(key)
        # Criar um nó para cada coluna 
        for col_name in df.columns:
            series = df[col_name]
            time_series_data = {}
            # Propriedades do nó serão data: valor
            for date_str, value in series.items():
                time_series_data[date_str] = value

            full_id = (obj.type + "_" + str(obj.code)+ "_"+col_name).replace(" ","")
            property_type = f"{key}_Dataframe"

            G.add_node(full_id, type=property_type ,object=obj)
            G.add_edge(node,full_id, title="Has_Property")
            print(G.nodes())
                        
            time_series_data['Full_ID'] = full_id
            time_series_data['ObjType'] = property_type
            time_series_data['Code'] = obj.code

            properties[full_id] = time_series_data

    except Exception as e:
        print(e)

    print(key, G.nodes())
    return G,properties 

def extract_node_properties(G_original):
    node_properties = {} 
    G = G_original.copy(as_view=False) 

    for n, attrs in G_original.nodes(data=True):

        # Criar grafo para adicionar nós de propriedades sem alterar 

        obj = attrs["object"]
        properties = {}

        # 1. Adiciona as propriedades básicas
        properties['Full_ID'] = n
        properties['ObjType'] = attrs['type']
        properties['Code'] = obj.code
        
        # Get descriptions (properties)
        try:
            node_properties[n] = properties
            data = obj.descriptions()
            filtered_data = {}
            for key in data.keys():
                # Condição: Se a chave NÃO começa com 'Ref' 
                if not obj.description(key).is_reference():

                    # Static properties 
                    if not obj.description(key).is_dynamic():
                        value = obj.get(key)

                        if isinstance(value,str):
                            value=value.rstrip()
                    
                        if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                            # Se o valor não é nativo, converte para string
                            try:
                                value = str(value)
                                value.rstrip()
                            except:
                                value = "Valor complexo não serializável"
                       
                        filtered_data[key] = value

                    else:
                        try: 
                            df = obj.get_df(key)
                            if df.empty:
                                filtered_data[key] = 0 
                            elif len(df)==1:
                                for col_name in df.columns:
                                    series= df[col_name]
                                    filtered_data[col_name]=series.iloc[0]
                            else:
                                for col_name in df.columns:
                                    series= df[col_name]
                                    if "01/01/1900" in df.index: 
                                        filtered_data[col_name]=series.loc["01/01/1900"] 

                                G,node_properties = extract_df_properties(obj,key,G, node_properties,n)

                        except Exception as e:
                            print(f"Erro ao processar propriedade dinâmica {key}: {e}")
                            continue

            properties.update(filtered_data)
        except Exception as e:
            print(f"Erro ao inspecionar atributos do nó {n}: {e}")

        # 3. Armazena as propriedades no dicionário Python
        node_properties[n] = properties


    return G, node_properties