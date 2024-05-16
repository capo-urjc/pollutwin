import requests
from bs4 import BeautifulSoup

import csv


def main():
    # URL de la página a la que queremos acceder (kms 14.41D, 17.42C, 23.0D)
    urls: list = [
        "https://infocar.dgt.es/etraffic/DetallesElementos?accion=detallesElemento&tipo=SensorTrafico&nombre=A-5%20Pk"
        "%2014.41%20D&elemGenCod=GUID_SEC_112382",
        "https://infocar.dgt.es/etraffic/DetallesElementos?accion=detallesElemento&tipo=SensorTrafico&nombre=A-5%20Pk"
        "%2017.42%20C&elemGenCod=GUID_SEC_112396",
        "https://infocar.dgt.es/etraffic/DetallesElementos?accion=detallesElemento&tipo=SensorTrafico&nombre=A-5%20Pk"
        "%2023.0%20D&elemGenCod=GUID_SEC_112413"]
    keys = ["14.41", "17.42", "23.0"]

    with open("data.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        data: dict = {}

        for url in urls:
            response = requests.get(url)

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')

                element = soup.find_all("li")

                km = str(soup.find("b")).split(' ')[2]

                values: list = []
                for e in element[1:-1]:
                    sub_str = e.text.split(':')

                    splitted_value = sub_str[1].split(' ')
                    values.append(splitted_value[1])
            else:
                print("Error al acceder a la URL:", response.status_code)

            data[km] = values

        measures: list = []
        for key in keys:
            values = data[key]

            if len(values) < 4:
                for i in range(4):
                    measures.append("NaN")

            else:
                for v in values:
                    measures.append(v)

        writer.writerow([measures[0]] + [measures[1]] + [measures[2]] + [measures[3]] + [measures[4]] + [measures[5]]
                        + [measures[6]] + [measures[7]] + [measures[8]] + [measures[9]]
                        + [measures[10]] + [measures[11]])


if __name__ == '__main__':
    main()
