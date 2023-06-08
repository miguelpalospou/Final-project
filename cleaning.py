import numpy as np
import pandas as pd
import re

def extraction():

    df_1 = pd.read_csv("data_scrapped/df_1.csv",index_col=False)
    df_2 = pd.read_csv("data_scrapped/df_2.csv",index_col=False)
    df_3 = pd.read_csv("data_scrapped/df_3.csv",index_col=False)
    df_4 = pd.read_csv("data_scrapped/df_4.csv",index_col=False)
    df_5 = pd.read_csv("data_scrapped/df_5.csv",index_col=False)
    df_6 = pd.read_csv("data_scrapped/df_6.csv",index_col=False)

    df = pd.concat([df_1,df_2,df_3,df_4,df_5,df_6], axis=0)
    return df


def cleaning(df):
    # renaming
    df = df.rename(columns={'district': 'neighbourhood'})

    # Here I'm creating a regex patern I will use to extract the neighborhoods from the column
    pattern = r"\b((?:Les Corts|Sant Andreu|El Raval|La Dreta de l'Eixample|El Gòtic|Sant Pere - Santa Caterina i la Ribera|La Nova Esquerra de l'Eixample|L'Antiga Esquerra de l'Eixample|La Sagrada Família|El Poble Sec - Parc de Montjuïc|Sant Antoni|El Camp d'En Grassot i Gràcia Nova|Vila de Gràcia|Diagonal Mar i el Front Marítim del Poblenou|Sants|El Guinardó|Pedralbes|El Putxet i el Farró|La Marina del Port|Sarrià|El Carmel|El Camp de l'Arpa del Clot|El Baix Guinardó|La Teixonera|Sant Gervasi - La Bonanova|Vilapicina i la Torre Llobeta|El Poblenou|Sant Martí de Provençals|Sants - Badal|El Besòs|El Fort Pienc|Les Tres Torres|La Trinitat Vella|Hostafrancs|La Verneda i la Pau|Vallcarca i els Penitents|Les Roquetes|La Barceloneta|El Congrés i els Indians|Ciutat Meridiana - Torre Baró - Vallbona|La Maternitat i Sant Ramon|La Prosperitat|Provençals del Poblenou|La Sagrera|La Salut|El Parc i la Llacuna del Poblenou|La Font de la Guatlla|Can Baró|La Bordeta|Can Peguera - El Turó de la Peira|El Clot|Navas|Horta|Porta|Vallvidrera - El Tibidabo i les Planes|El Bon Pastor|La Font d'En Fargues|La Vila Olímpica del Poblenou|Sant Genís Dels Agudells - Montbau|Verdun|El Coll|La Trinitat Nova|La Marina del Prat Vermell|Sant Gervasi - Galvany|La Guineueta|La Vall d'Hebron - La Clota|Canyelles|Baró de Viver|Zona Franca - Port))\b"
    df['neighbourhood'] = df['neighbourhood'].str.extract(pattern)

    df = df.drop('Unnamed: 0', axis=1)

    # Using a dictionary to group the neighborhoods into their corresponding districts
    districts = {
    "Ciutat Vella": ["El Raval", "El Gòtic", "La Barceloneta", "Sant Pere - Santa Caterina i la Ribera"],
    "Eixample": ["El Fort Pienc", "La Sagrada Família", "La Dreta de l'Eixample", "L'Antiga Esquerra de l'Eixample", "La Nova Esquerra de l'Eixample", "Sant Antoni"],
    "Sants-Montjuïc": ["La Marina del Port","El Poble Sec", "La Marina del Prat Vermell", "La Marina de Port", "La Font de la Guatlla", "Hostafrancs", "La Bordeta", "Sants-Badal", "Sants", "El Poble Sec - Parc de Montjuïc"],
    "Les Corts": ["Les Corts", "La Maternitat i Sant Ramon", "Pedralbes"],
    "Sarrià-Sant Gervasi": ["Vallvidrera - El Tibidabo i les Planes", "Sarrià", "Les Tres Torres", "Sant Gervasi - La Bonanova", "Sant Gervasi - Galvany", "El Putxet i el Farró"],
    "Gràcia": ["Vallcarca i els Penitents", "El Coll", "La Salut", "Vila de Gràcia", "El Camp d'En Grassot i Gràcia Nova"],
    "Horta-Guinardó": ["El Baix Guinardó", "Can Baró", "El Guinardó", "La Font d'En Fargues", "El Carmel", "La Teixonera", "Sant Genís Dels Agudells - Montbau", "La Vall d'Hebron - La Clota", "Horta"],
    "Nou Barris": ["Ciutat Meridiana - Torre Baró - Vallbona","Vilapicina i la Torre Llobeta", "Porta", "El Turó de la Peira", "Can Peguera - El Turó de la Peira", "La Guineueta", "Canyelles", "Les Roquetes", "Verdun", "La Prosperitat", "La Trinitat Nova", "Torre Baró", "Ciutat Meridiana", "Vallbona"],
    "Sant Andreu": ["El Besòs","La Trinitat Vella", "Baró de Viver", "El Bon Pastor", "Sant Andreu", "La Sagrera", "El Congrés i els Indians", "Navas"],
    "Sant Martí": ["El Camp de l'Arpa del Clot", "El Clot", "El Parc i la Llacuna del Poblenou", "La Vila Olímpica del Poblenou", "El Poblenou", "Diagonal Mar i el Front Marítim del Poblenou", "El Besòs i el Maresme", "Provençals del Poblenou", "Sant Martí de Provençals", "La Verneda i la Pau"]
    }

    df['district'] = df['neighbourhood'].map({neighbourhood: district for district, neighbourhoods in districts.items() for neighbourhood in neighbourhoods})

    # dropping useless columns and removing symbols, acronyms, etc.
    df = df.drop('street', axis=1)
    df = df.drop('description', axis=1)    

    df['price'] = df['price'].str.replace('€', '')
    df['area'] = df['area'].str.replace('m²', '')
    df['rooms'] = df['rooms'].str.replace('bed.', '')
    df['plant'] = df['plant'].str.replace('th', '')
    df['plant'] = df['plant'].str.replace('rd', '')
    df['plant'] = df['plant'].str.replace('st', '')
    df['plant'] = df['plant'].str.replace('Ground', '0')
    df['plant'] = df['plant'].str.replace('nd', '')

    df = df.drop(df[df["plant"] == "Mezzanine"].index)
    df = df.drop(df[df["area"] == "4th floor exterior without lift"].index)
    df = df.drop(df[df["area"] == "Ground floor exterior without lift"].index)
    df = df.drop(df[df["area"] == "6th floor exterior with lift"].index)
    df = df.drop(df[df["area"] == "1st floor exterior with lift"].index)
    df = df.drop(df[df["area"] == " exterior without lift"].index)

    df['price'] = df['price'].str.replace(',', '')
    df['area'] = df['area'].str.replace(',', '')

    # converting to numeric
    df['area'] = df['area'].astype(int)
    df['price'] = df['price'].astype(int)
    df['rooms'] = df['rooms'].astype(int)
    df['plant'] = pd.to_numeric(df['plant'], errors='coerce')   

    # formatting parking column
    df['parking'] = df['parking'].fillna('no')
    df= df[(df.parking == "no") | (df.parking == "Parking included")]
    df = df.drop_duplicates(subset='reference')
    df['parking'] = df['parking'].str.replace("Parking included", "yes")

    # formatting lift column
    df['lift'] = df['lift'].str.replace("with lift", "lift")
    df['lift'] = df['lift'].str.replace("without lift", "no lift")
    df['lift'] = df['lift'].fillna('no lift')

    # cleaning type column

    df['type'] = df['type'].str.replace("Detached", "House")
    df = df.drop(df[df['type'] == 'Tower'].index)
    df = df.drop(df[df['type'] == 'Cortijo'].index)

    df.to_csv("df_final.csv")

    return df

def tableau (df):
    neighborhoods = {
    'La Dreta de l\'Eixample': 'la Dreta de l\'Eixample',
    'El Gòtic': 'el Barri Gòtic',
    'L\'Antiga Esquerra de l\'Eixample': 'l\'Antiga Esquerra de l\'Eixample',
    'Sant Gervasi - Galvany': 'Sant Gervasi - Galvany',
    'Sant Pere - Santa Caterina i la Ribera': 'Sant Pere, Santa Caterina i la Ribera',
    'Diagonal Mar i el Front Marítim del Poblenou': 'Diagonal Mar i el Front Marítim del Poblenou',
    'El Raval': 'el Raval',
    'Sant Gervasi - La Bonanova': 'Sant Gervasi - la Bonanova',
    'Pedralbes': 'Pedralbes',
    'La Sagrada Família': 'la Sagrada Família',
    'La Nova Esquerra de l\'Eixample': 'la Nova Esquerra de l\'Eixample',
    'Vila de Gràcia': 'la Vila de Gràcia',
    'Sants': 'Sants',
    'El Putxet i el Farró': 'el Putxet i el Farró',
    'Sant Antoni': 'Sant Antoni',
    'El Poble Sec - Parc de Montjuïc': 'el Poble-sec',
    'Sarrià': 'Sarrià',
    'La Vila Olímpica del Poblenou': 'la Vila Olímpica del Poblenou',
    'El Camp d\'En Grassot i Gràcia Nova': 'el Camp d\'en Grassot i Gràcia Nova',
    'Les Tres Torres': 'les Tres Torres',
    'El Camp de l\'Arpa del Clot': 'el Camp de l\'Arpa del Clot',
    'La Salut': 'la Salut',
    'El Fort Pienc': 'el Fort Pienc',
    'Vallvidrera - El Tibidabo i les Planes': 'Vallvidrera, el Tibidabo i les Planes',
    'El Poblenou': 'el Poblenou',
    'El Congrés i els Indians': 'el Congrés i els Indians',
    'El Guinardó': 'el Guinardó',
    'La Maternitat i Sant Ramon': 'la Maternitat i Sant Ramon',
    'El Baix Guinardó': 'el Baix Guinardó',
    'Vallcarca i els Penitents': 'Vallcarca i els Penitents',
    'La Barceloneta': 'la Barceloneta',
    'Vilapicina i la Torre Llobeta': 'Vilapicina i la Torre Llobeta',
    'Hostafrancs': 'Hostafrancs',
    'La Font d\'En Fargues': 'la Font d\'en Fargues',
    'Horta': 'Horta',
    'El Parc i la Llacuna del Poblenou': 'el Parc i la Llacuna del Poblenou',
    'El Carmel': 'el Carmel',
    'La Bordeta': 'la Bordeta',
    'La Teixonera': 'la Teixonera',
    'La Font de la Guatlla': 'la Font de la Guatlla',
    'La Sagrera': 'la Sagrera',
    'Navas': 'Navas',
    'Provençals del Poblenou': 'Provençals del Poblenou',
    'El Besòs': 'el Besòs i el Maresme',
    'El Coll': 'el Coll',
    'Ciutat Meridiana - Torre Baró - Vallbona': 'Ciutat Meridiana',
    'La Marina del Port': 'la Marina de Port',
    'Can Baró': 'Can Baró',
    'Sant Martí de Provençals': 'Sant Martí de Provençals',
    'El Clot': 'el Clot',
    'Porta': 'Porta',
    'La Verneda i la Pau': 'la Verneda i la Pau',
    'Can Peguera - El Turó de la Peira': 'el Turó de la Peira',
    'El Bon Pastor': 'el Bon Pastor',
    'La Prosperitat': 'la Prosperitat',
    'Sant Genís Dels Agudells - Montbau': 'Sant Genís dels Agudells',
    'Les Roquetes': 'les Roquetes',
    'La Trinitat Vella': 'la Trinitat Vella',
    'La Guineueta': 'la Guineueta',
    'La Vall d\'Hebron - La Clota': 'la Vall d\'Hebron',
    'Verdun': 'Verdun',
    'La Trinitat Nova': 'la Trinitat Nova',
    'Les Corts': 'les Corts',
    'Sant Andreu': 'Sant Andreu'
    
    }

    df_tableau1=df
    df_tableau1['neighbourhood'] = df_tableau1['neighbourhood'].map(neighborhoods)
    df_tableau1.to_csv("data/df_tableau1.csv")
    return df



