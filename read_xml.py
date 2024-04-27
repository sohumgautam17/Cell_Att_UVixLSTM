import xml.etree.ElementTree as ET
from PIL import Image

# Path to your files
xml_file = './Annotations/TCGA-18-5592-01Z-00-DX1.xml'
tree = ET.parse(xml_file)
root = tree.getroot()

# Dictionary to hold all annotation data
annotations = []

# Iterate through each Annotation in the XML
for annotation in root.findall('Annotation'):
    annotation_data = {
        'Id': annotation.get('Id'),
        'Type': annotation.get('Type'),
        'LineColor': annotation.get('LineColor'),
        'Regions': []
    }

    # Access Regions within each Annotation
    for region in annotation.findall('.//Region'):
        region_data = {
            'Id': region.get('Id'),
            'Type': region.get('Type'),
            'Length': region.get('Length'),
            'Area': region.get('Area'),
            'Vertices': []
        }

        # Access Vertices within each Region
        for vertex in region.findall('.//Vertex'):
            vertex_data = {
                'X': vertex.get('X'),
                'Y': vertex.get('Y')
            }
            region_data['Vertices'].append(vertex_data)

        annotation_data['Regions'].append(region_data)

    annotations.append(annotation_data)

# Print or process the extracted data
for annotation in annotations:
    print(annotation)
    