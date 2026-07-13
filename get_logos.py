import urllib.request
from bs4 import BeautifulSoup

pages = ['Partido_Socialista_Obrero_Espa%C3%B1ol', 'Podemos', 'Euskal_Herria_Bildu', 'Esquerra_Republicana_de_Catalunya', 'Vox_(partido_pol%C3%ADtico)', 'Sumar_(coalici%C3%B3n)', 'Partido_Nacionalista_Vasco']
for p in pages:
    req = urllib.request.Request(f'https://es.wikipedia.org/wiki/{p}', headers={'User-Agent': 'Mozilla/5.0'})
    try:
        html = urllib.request.urlopen(req).read().decode()
        soup = BeautifulSoup(html, 'html.parser')
        infobox = soup.find('table', class_='infobox')
        if infobox:
            img = infobox.find('img')
            if img:
                print(f'{p}: https:{img["src"]}')
            else:
                print(f'{p}: No img tag in infobox')
        else:
            print(f'{p}: No infobox')
    except Exception as e:
        print(f'{p}: ERROR {e}')
