import streamlit as st
import numpy as np
import pandas as pd
import pickle 


st.set_page_config(layout='centered')
st.title('Predict cars price in Egyptian Market')
st.write(''' On this page, I am using a machine learning model to predict the price of any car according to data entered by the user. Data has been scraped from the website [Hatla2e](https://eg.hatla2ee.com/en)
The dataset contains **31,194** used car Ads, each ad. Consists of a description of the car brand, model, date of the Ad., model year, traveled kilometers, color, and price.
Dataset has been cleaned and wrangled. Outliers have been removed and finally, the best machine-learning model has been selected.
''')

st.write(''' **Note:** I have made every effort to ensure the accuracy and reliability the machine learning model. However,
  I don’t give any kind of warranty and am not responsible for any decision taken by the user.''' )

st.sidebar.image('Photo.png',width=60)
st.sidebar.subheader('Developed by [Amr Balbaa](https://www.linkedin.com/in/amr-balbaa/)')
st.sidebar.write('If you have any constructive feedback please reach out to this email amr.balbaa@gmail.com or click this [link](www.linkedin.com/in/amr-balbaa)')




# Dictionaries of label encoding columns
# Brand encoding
brand_dic = {'Alfa Romeo': 0, 'Aston Martin': 1, 'Audi': 2, 'BMW': 3, 'Baic': 4, 'Bestune': 5, 'Brilliance': 6, 'Byd': 7, 'Cadillac': 8, 'Canghe  || Changhe': 9, 'Canghe || Changhe': 10, 'Chana': 11, 'Changan': 12, 'Chery': 13, 'Chevrolet': 14, 'Chrysler': 15, 'Citroën': 16, 'Cupra': 17, 'DFSK': 18, 'Daewoo': 19, 'Daihatsu': 20, 'Dodge': 21, 'Domy': 22, 'Dongfeng': 23, 'Ds': 24, 'Emgrand': 25, 'Exeed': 26, 'Faw': 27, 'Fiat': 28, 'Ford': 29, 'Foton': 30, 'Geely': 31, 'Gmc': 32, 'Great Wall': 33, 'Hafei': 34, 'Haima': 35, 'Haval': 36, 'Hawtai': 37, 'Honda': 38, 'Hummer': 39, 'Hyundai': 40, 'Infiniti': 41, 'Isuzu': 42, 'Jac': 43, 'Jaguar': 44, 'Jeep': 45, 'Jetour': 46, 'Kaiyi': 47, 'Karry': 48, 'Kenbo': 49, 'Keyton': 50, 'Kia': 51, 'Lada': 52, 'Lancia': 53, 'Land Rover': 54, 'Landwind': 55, 'Lexus': 56, 'Lifan': 57, 'Lincoln': 58, 'MG': 59, 'Mahindra': 60, 'Maserati': 61, 'Mazda': 62, 'Mercedes': 63, 'Mini': 64, 'Mitsubishi': 65, 'Nissan': 66, 'Opel': 67, 'Peugeot': 68, 'Pontiac': 69, 'Porsche': 70, 'Proton': 71, 'Renault': 72, 'Saipa': 73, 'Seat': 74, 'Senova': 75, 'Skoda': 76, 'Smart': 77, 'Sokon': 78, 'Soueast': 79, 'Speranza': 80, 'Ssang Yong': 81, 'Subaru': 82, 'Suzuki': 83, 'Toyota': 84, 'Victory': 85, 'Volkswagen': 86, 'Volvo': 87, 'Zotye': 88}

# Model Encoding dic
model_dic = {'1 Series': 0, '11': 1, '116': 2, '118': 3, '120': 4, '1200': 5, '121': 6, '128': 7, '1300': 8, '133': 9, '146': 10, '1500': 11,   '156': 12, '16': 13, '18': 14, '180': 15, '19': 16, '2 || Mazda 2': 17, '200': 18, '2008': 19, '2017': 20, '206': 21, '207': 22, '208': 23, '2101': 24, '2105': 25, '2107': 26, '2110': 27, '2111': 28, '218 i': 29, '220': 30, '225': 31, '232': 32, '240': 33, '250': 34, '280': 35, '3': 36, '3  || Mazda 3': 37, '3 || Mazda 3': 38, '300': 39, '3008': 40, '301': 41, '306': 42, '307': 43, '308': 44, '308 sw': 45, '316': 46, '318': 47, '320': 48, '323': 49, '325': 50, '328': 51, '330': 52, '330S': 53, '350': 54, '360 || MG 360': 55, '405': 56, '406': 57, '407': 58, '408': 59, '460': 60, '5': 61, '5 Series': 62, '500': 63, '500 E': 64, '500 X': 65, '5008': 66, '500C': 67, '505': 68, '508': 69, '520': 70, '523': 71, '525': 72, '528': 73, '530': 74, '535': 75, '6': 76, '607': 77, '620': 78, '645': 79, '650': 80, '735': 81, '740': 82, '750': 83, '80': 84, '9': 85, 'A 10 || A10': 86, 'A 150': 87, 'A 160': 88, 'A 180': 89, 'A 200': 90, 'A 250': 91, 'A1': 92, 'A11': 93, 'A113': 94, 'A13': 95, 'A15': 96, 'A213': 97, 'A25': 98, 'A3': 99, 'A4': 100, 'A5': 101, 'A516': 102, 'A6': 103, 'A620 || A620': 104, 'A9': 105, 'AMG GT': 106, 'APV': 107, 'Accent': 108, 'Accent HCI': 109, 'Accent RB': 110, 'Accord': 111, 'Adam': 112, 'Aeolus A30': 113, 'Albea': 114, 'Alsvin': 115, 'Altea XL': 116, 'Alto': 117, 'Amarok': 118, 'Armada': 119, 'Arona': 120, 'Arrizo 5': 121, 'Arteon': 122, 'Astra': 123, 'Astro': 124, 'Ateca': 125, 'Atos': 126, 'Attrage': 127, 'Auris': 128, 'Avante': 129, 'Avanza': 130, 'Avensis': 131, 'Aveo': 132, 'Ax': 133, 'Azera': 134, 'Azkarra': 135, 'B 150': 136, 'B 160': 137, 'B 180': 138, 'B 200': 139, 'B Class || B Class': 140, 'B-Max': 141, 'B15': 142, 'Baleno': 143, 'Bayon': 144, 'Beetle': 145, 'Belta': 146, 'Benni': 147, 'Berlingo': 148, 'Besturn': 149, 'Bingo': 150, 'Blazer': 151, 'Bora': 152, 'Brava': 153, 'Bravo': 154, 'C 180': 155, 'C 200': 156, 'C 200 AMG': 157, 'C 230': 158, 'C 240': 159, 'C 250': 160, 'C 280': 161, 'C 300': 162, 'C 350': 163, 'C Elysee': 164, 'C-HR': 165, 'C-Max': 166, 'C1': 167, 'C3': 168, 'C3 Aircross': 169, 'C30': 170, 'C31': 171, 'C4': 172, 'C4 Grand Picasso': 173, 'C4 Picasso': 174, 'C5': 175, 'C5 Aircross': 176, 'C8': 177, 'CC': 178, 'CLA 180': 179, 'CLA 200': 180, 'CLK': 181, 'CLS Class': 182, 'CRV': 183, 'CRX': 184, 'CS 15': 185, 'CS 35': 186, 'CS 35 Plus': 187, 'CS 55': 188, 'Caddy': 189, 'Cadenza': 190, 'Caliber': 191, 'Camaro': 192, 'Camry': 193, 'Captiva': 194, 'Captur': 195, 'Caravan': 196, 'Carens': 197, 'Carnival': 198, 'Carry': 199, 'Cascada': 200, 'Cavalier': 201, 'Cayenne': 202, 'Cayenne S': 203, 'Ceed': 204, 'Celerio': 205, 'Cerato': 206, 'Cerato Koup': 207, 'Challenger': 208, 'Charade': 209, 'Charger': 210, 'Cherokee': 211, 'Chery QQ || Chery qq': 212, 'Ciaz': 213, 'Cielo': 214, 'Cinquecento': 215, 'City': 216, 'Civic': 217, 'Ck': 218, 'Ck2': 219, 'Clio': 220, 'Clubman': 221, 'Colt': 222, 'Commander': 223, 'Compass': 224, 'Convertible': 225, 'Cool Ray': 226, 'Cooper': 227, 'Cooper F55': 228, 'Cooper Roadster': 229, 'Cordoba': 230, 'Corolla': 231, 'Corolla Cross': 232, 'Corona': 233, 'Corsa': 234, 'Corvette': 235, 'Country man': 236, 'Countryman S': 237, 'Coupe': 238, 'Creta': 239, 'Creta SU2 || Creta': 240, 'Cross': 241, 'Crossland': 242, 'Cruze': 243, 'Crystala': 244, 'Cx 3': 245, 'D max': 246, 'DB9': 247, 'DS5': 248, 'DS7': 249, 'DX3': 250, 'DX5': 251, 'Dart': 252, 'Dedra': 253, 'Defender': 254, 'Discovery': 255, 'Doblo': 256, 'Dogan': 257, 'Ds4': 258, 'Ds5': 259, 'Dts': 260, 'Ducato': 261, 'Durango': 262, 'Duster': 263, 'Duster Nova': 264, 'Dx7 prime': 265, 'E': 266, 'E 200': 267, 'E 200 AMG': 268, 'E 230': 269, 'E 240': 270, 'E 250': 271, 'E 280': 272, 'E 300': 273, 'E 320': 274, 'E 350': 275, 'E Golf': 276, 'E5': 277, 'EAGLE 580': 278, 'EC 7': 279, 'EON': 280, 'EX7': 281, 'EX80': 282, 'Eado': 283, 'Eado plus': 284, 'Eagle': 285, 'Eagle pro': 286, 'Echo': 287, 'Eclipse': 288, 'Eclipse Cross': 289, 'EcoSport': 290, 'Edge': 291, 'Elantra': 292, 'Elantra AD': 293, 'Elantra CN7': 294, 'Elantra Coupe': 295, 'Elantra HD': 296, 'Elantra MD': 297, 'Emgrand': 298, 'Emgrand 7': 299, 'Emgrand X7': 300, 'Englon': 301, 'Envoy': 302, 'Envy': 303, 'Eos': 304, 'Equinox': 305, 'Ertiga': 306, 'Escort': 307, 'Espero': 308, 'Excel': 309, 'Expedition': 310, 'Explorer': 311, 'F-150': 312, 'F0': 313, 'F3': 314, 'F3R': 315, 'FRV': 316, 'FRV Cross': 317, 'FSV': 318, 'Fabia': 319, 'Familia': 320, 'Family': 321, 'Fantasia': 322, 'Favorit': 323, 'Felicia': 324, 'Felicia combi': 325, 'Fiesta': 326, 'Fiorino': 327, 'Florida': 328, 'Fluence': 329, 'Flyer': 330, 'Focus': 331, 'Foreste': 332, 'Formentor': 333, 'Forte': 334, 'Fortuner': 335, 'Fortwo': 336, 'Fox': 337, 'Freestar': 338, 'Frontera': 339, 'Fruits': 340, 'Fusion': 341, 'Fx': 342, 'G6': 343, 'GLA': 344, 'GLA 180': 345, 'GLK': 346, 'GLK 250': 347, 'GLK 350': 348, 'Galant': 349, 'Galena': 350, 'Galloper': 351, 'Gen 2': 352, 'Genesis': 353, 'Getz': 354, 'Ghibli': 355, 'Giulia': 356, 'Giulietta': 357, 'Glory': 358, 'Glory 330': 359, 'Glory Van': 360, 'Gol': 361, 'Golf': 362, 'Gran max': 363, 'Grand C-Max': 364, 'Grand C4 Spacetourer': 365, 'Grand Cerato': 366, 'Grand Cherokee': 367, 'Grand Santa Fe': 368, 'Grand i10': 369, 'Grand marquis': 370, 'Grand terios': 371, 'Grandis': 372, 'Grandland': 373, 'Granta': 374, 'H 3': 375, 'H1': 376, 'H100': 377, 'H2': 378, 'H3': 379, 'H6': 380, 'HRV': 381, 'HS': 382, 'Haima 1': 383, 'Haima 2': 384, 'Haima 3': 385, 'Hiace': 386, 'Hilux': 387, 'Hover': 388, 'I10': 389, 'I20': 390, 'I3': 391, 'I30': 392, 'IX 35': 393, 'Ibiza': 394, 'Imperial': 395, 'Impreza': 396, 'Insignia': 397, 'Ix20': 398, 'J2': 399, 'J3': 400, 'J6': 401, 'J7': 402, 'JAZZ': 403, 'JS3': 404, 'JS4': 405, 'Jetta': 406, 'Jimmy': 407, 'Jimny': 408, 'Jinbei': 409, 'John Cooper': 410, 'Jolion': 411, 'Journey': 412, 'Juke': 413, 'Juliet': 414, 'Jumper': 415, 'K01': 416, 'K01s': 417, 'K5': 418, 'K8': 419, 'Ka': 420, 'Kadjar': 421, 'Kalina': 422, 'Kamiq': 423, 'Kancil': 424, 'Kangoo': 425, 'Karoq': 426, 'Kodiaq': 427, 'Komodo': 428, 'Kona': 429, 'Korando': 430, 'Kuga': 431, 'L3': 432, 'LEAF': 433, 'Lacetti': 434, 'Laguna': 435, 'Lancer': 436, 'Lancer Crystala': 437, 'Lancer EX Shark': 438, 'Lancer Puma': 439, 'Land Cruiser': 440, 'Lanos': 441, 'Lanos 2': 442, 'Lebron': 443, 'Legacy': 444, 'Leganza': 445, 'Leon': 446, 'Leon-cupra || LEON CUPR': 447, 'Liana': 448, 'Liberty': 449, 'Linea': 450, 'Livina': 451, 'Lobo': 452, 'Lodgy': 453, 'Logan': 454, 'Lr2': 455, 'M11': 456, 'M12': 457, 'M20': 458, 'M3': 459, 'M4': 460, 'M50  || M 50': 461, 'M50 || M 50': 462, 'M60': 463, 'M70': 464, 'MCV || MCV': 465, 'ML': 466, 'MZ 40': 467, 'Malibu': 468, 'Maple': 469, 'Marea': 470, 'Maruti': 471, 'Materia': 472, 'Matiz': 473, 'Matrix': 474, 'Maxima': 475, 'Maybach S500': 476, 'Megane': 477, 'Meriva': 478, 'Microbus': 479, 'Mini Cooper S': 480, 'Mini Truck': 481, 'Mini Van': 482, 'Minivan': 483, 'Mira': 484, 'Mirage': 485, 'Mito': 486, 'Mk': 487, 'Mohave': 488, 'Mokka': 489, 'Mondeo': 490, 'Montero': 491, 'Murano': 492, 'Musso': 493, 'Mx-5': 494, 'N-Series': 495, 'N200': 496, 'N300': 497, 'N5': 498, 'Navigator': 499, 'Neon': 500, 'New Star': 501, 'Niva': 502, 'Nova': 503, 'Nubira 1': 504, 'Nubira 2': 505, 'Octavia': 506, 'Octavia A4': 507, 'Octavia A5': 508, 'Octavia A7': 509, 'Octavia A8': 510, 'Octavia Combi': 511, 'Oka': 512, 'Opirus': 513, 'Optima': 514, 'Optra': 515, 'Outlander': 516, 'PT Cruiser': 517, 'PX Cargo': 518, 'Pacifica': 519, 'Pajero': 520, 'Palio': 521, 'Panda': 522, 'Pandino': 523, 'Parati': 524, 'Partner': 525, 'Passat': 526, 'Patrol': 527, 'Pegas': 528, 'Peri': 529, 'Persona': 530, 'Petra': 531, 'Picanto': 532, 'Pick up': 533, 'Pickup': 534, 'Pilot': 535, 'Pointer': 536, 'Polo': 537, 'Prado': 538, 'Preve': 539, 'Previa': 540, 'Pride': 541, 'Punto': 542, 'Punto evo': 543, 'Q2': 544, 'Q22': 545, 'Q3': 546, 'Q30': 547, 'Q5': 548, 'Q7': 549, 'QX': 550, 'Qashqai': 551, 'Qubo': 552, 'RCZ': 553, 'RX': 554, 'Rainbow': 555, 'Ram': 556, 'Range Rover': 557, 'Range Rover Evoque || Evoque': 558, 'Range Rover Vogue || Vogue': 559, 'Ranger': 560, 'Rapid': 561, 'Rapid Spaceback': 562, 'Rav 4': 563, 'Renegade': 564, 'Rio': 565, 'Rogue': 566, 'Roomster': 567, 'Royal': 568, 'Rumion': 569, 'Rush': 570, 'Rx': 571, 'Rx5': 572, 'S 320': 573, 'S 350': 574, 'S 500': 575, 'S Class': 576, 'S Presso': 577, 'S-Type': 578, 'S2': 579, 'S3': 580, 'S30': 581, 'S4': 582, 'S40': 583, 'S5': 584, 'S60': 585, 'S7': 586, 'S80': 587, 'SEL 500': 588, 'SIRION': 589, 'SLK': 590, 'SLK Class': 591, 'Saga': 592, 'Saipa': 593, 'Samara': 594, 'Sandero': 595, 'Sandero Step Way': 596, 'Santa Fe': 597, 'Santamo': 598, 'Scala': 599, 'Scenic': 600, 'Scorpio': 601, 'Sebring': 602, 'Seltos': 603, 'Sentra': 604, 'Sephia': 605, 'Sequoia': 606, 'Shahin': 607, 'Shuma': 608, 'Siena': 609, 'Solaris': 610, 'Sonata': 611, 'Sonic': 612, 'Sorento': 613, 'Soul': 614, 'Space Star': 615, 'Space Wagon': 616, 'Spark': 617, 'Spark Lite': 618, 'Sparky': 619, 'Spectra': 620, 'Splendor': 621, 'Sportage': 622, 'Stratus': 623, 'Suburban': 624, 'Sunny': 625, 'Superb': 626, 'Suran': 627, 'Swift': 628, 'Swift Dzire || Dzire': 629, 'Sx4': 630, 'Symbol': 631, 'T-Series': 632, 'T500': 633, 'T55': 634, 'T600': 635, 'T77 Pro': 636, 'TXL': 637, 'Tahoe || Taho': 638, 'Talisman': 639, 'Tarraco': 640, 'Taurus': 641, 'Tempra': 642, 'Tercel': 643, 'Terios': 644, 'Terrain': 645, 'Thunderbird': 646, 'Tiba': 647, 'Tiburon': 648, 'Tico': 649, 'Tiggo': 650, 'Tiggo 2': 651, 'Tiggo 3': 652, 'Tiggo 4': 653, 'Tiggo 7': 654, 'Tiggo 7 pro': 655, 'Tiggo 8': 656, 'Tiggo 8 Pro': 657, 'Tigra': 658, 'Tiguan': 659, 'Tiida': 660, 'Tipo': 661, 'Tivoli': 662, 'Tivoli XLV': 663, 'Toledo': 664, 'Tons 4': 665, 'Touareg': 666, 'Town': 667, 'Town & Country': 668, 'Trajet': 669, 'Transit': 670, 'Transporter': 671, 'Trax': 672, 'Trooper': 673, 'Tucson': 674, 'Tucson GDI': 675, 'Tucson Turbo GDI  || Tucson Turbo': 676, 'Tucson Turbo GDI || Tucson Turbo': 677, 'Turan': 678, 'Uno': 679, 'Uplander': 680, 'Urvan': 681, 'V 60': 682, 'V1': 683, 'V10': 684, 'V101': 685, 'V3': 686, 'V5': 687, 'V6': 688, 'V7': 689, 'Van': 690, 'Vectra': 691, 'Veloster': 692, 'Verna': 693, 'Veryca': 694, 'Viano': 695, 'View': 696, 'Vita': 697, 'Vitara': 698, 'Viva': 699, 'Voyager': 700, 'Waja': 701, 'Wingle 5': 702, 'Wira': 703, 'Wrangler': 704, 'Wrangler Unlimited': 705, 'X 30': 706, 'X Pandino': 707, 'X-Type': 708, 'X1': 709, 'X2': 710, 'X25': 711, 'X3': 712, 'X35': 713, 'X4': 714, 'X40': 715, 'X5': 716, 'X7': 717, 'X7 || BEIJING X7': 718, 'X70': 719, 'X70 Plus': 720, 'X70S': 721, 'X95': 722, 'XC 40': 723, 'XC90': 724, 'XD': 725, 'XE': 726, 'XF': 727, 'XJ': 728, 'XTrail': 729, 'XV': 730, 'Xceed': 731, 'Xpander': 732, 'Xplosion': 733, 'Xsara': 734, 'YRV': 735, 'Yaris': 736, 'Yeti': 737, 'Z100': 738, 'Z200': 739, 'ZS': 740, 'ZX': 741, 'Zafira Tourer': 742, 'Zaz': 743, 'foison || Foison': 744, 'ideal || Ideal': 745, 'lr3': 746}


# Transmission Type dic
transmission_type_dic = {'automatic': 1, 'manual': 2}
# color label encoding
color_dic = {'Beige': 0, 'Black': 1, 'Blue': 2, 'Bronze': 3, 'Brown': 4, 'Champagne': 5, 'Cyan': 6, 'Dark blue': 7, 'Dark green': 8, 'Dark red': 9, 'Eggplant': 10, 'Gold': 11, 'Gray': 12, 'GrayGlory': 13, 'Green': 14, 'Light grey': 15, 'Mocha': 16, 'Olive': 17, 'Orange': 18, 'Petroleum': 19, 'Purple': 20, 'Red': 21, 'Silver': 22, 'White': 23, 'Yellow': 24}



# Import cars dataset

df = pd.read_csv('cars_cleaned_data.csv',index_col=0)

st.image('car2.jpg',width=750)


col1, col2 =  st.columns(2)

# Brand selectbox
brand = df['brand'].unique()
brand_input = col1.selectbox(' Car brand',brand)

# function to select the models that matched the brand
def get_model(brand):
    models = df[df['brand']==brand]['model'].unique() 
    return models 

model_input = col2.selectbox('Car Model',get_model(brand_input))

# Transmission type selectbox
tt = df.groupby(['model','transmission_type']).mean().loc[model_input].reset_index(level=0)['transmission_type'].unique()
Transmission_Type_input = col1.selectbox('Transmission Type',tt)

# Color select box
#color = df['color'].unique()
cc = df.groupby(['model','color']).mean().loc[model_input].reset_index(level=0)['color'].unique()
color_input = col2.selectbox('Car Colour',cc)

# Model Year
min_year = df[df['model'] == model_input]['year'].min()
max_year = df[df['model'] == model_input]['year'].max()
year_input = col1.slider('Model Year',int(min_year),int(max_year))

# total kilometers traveled
kilometer_input = col2.number_input('Enter total kilometers traveled') 







# Collect user inputs

def user_inputs_prediction():
    selected_brand = brand_dic[brand_input]
    selected_model = model_dic[model_input]
    selected_trans_type = transmission_type_dic[Transmission_Type_input]
    selected_color = color_dic[color_input]
    selected_year = year_input
    selected_kilometers = kilometer_input
    user_data = np.array([[selected_brand,
                            selected_model,
                            selected_trans_type,
                            selected_color,
                            selected_year,
                            selected_kilometers]])
    pickled_model = pickle.load(open('ERF_model.pkl', 'rb'))

    return pickled_model.predict(user_data)




# Prediction button
predict_button = st.button('Predict')


if predict_button:
    try :
        price = round(float(user_inputs_prediction()))
        st.subheader('{} Thousand Egyptian Pounds'.format(price))
        st.success('Good Job')
    except:
        st.warning('Warning: Something went wrong, Please check entered data')

