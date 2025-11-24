import json

def extract_node_properties(G):
    node_properties = {} 

    for n, attrs in G.nodes(data=True):
        obj = attrs["object"]
        properties = {}

        # 1. Adiciona as propriedades básicas que você já tinha ou deseja fixar
        properties['Full_ID'] = n
        properties['Type'] = attrs['type']
        
        # Tentativa de pegar o Code/ID de forma mais limpa (assumindo o formato Type_Code)
        properties['Code'] = obj.code
        
        # Tentativa 1: Usar o método .to_dict() se ele existir (é o mais seguro)
        try:
            data = obj.as_dict()
            filtered_data = {}
            for key, value in data.items():
                # Condição: Se a chave NÃO começa com 'Ref' (case sensitive)
                if not key.startswith('Ref'):
                    final_value = value

                    if isinstance(final_value,str):
                        final_value=final_value.rstrip()
                    
                    if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                        # Se o valor não é nativo, converte para string
                        try:
                            final_value = str(value)
                        except:
                            final_value = "[Valor complexo não serializável]"
                            
                    filtered_data[key] = final_value

            properties.update(filtered_data)
        except Exception as e:
            print(f"Erro ao inspecionar atributos do nó {n}: {e}")

        # 3. Armazena as propriedades no dicionário Python
        node_properties[n] = properties

    return node_properties