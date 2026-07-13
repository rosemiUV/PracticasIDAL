import json
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

DB_PATH = Path(__file__).parent / "diputados_db.json"

async def extract_deputy(context, cod):
    url = f"https://www.congreso.es/es/busqueda-de-diputados?p_p_id=diputadomodule&p_p_lifecycle=0&p_p_state=normal&p_p_mode=view&_diputadomodule_mostrarFicha=true&codParlamentario={cod}&idLegislatura=XV&mostrarAgenda=false"
    page = await context.new_page()
    try:
        await page.goto(url, timeout=25000, wait_until="domcontentloaded")
        # Esperamos a que la SPA renderice la clase nombre-diputado
        await page.wait_for_selector(".nombre-diputado", timeout=10000)
        
        nombre_loc = page.locator(".nombre-diputado")
        if await nombre_loc.count() == 0:
            await page.close()
            return None
            
        nombre = (await nombre_loc.first.inner_text()).strip()
        if not nombre:
            await page.close()
            return None
            
        # Extraer foto
        img_src = ""
        foto_loc = page.locator("#foto-diputado")
        if await foto_loc.count() > 0:
            img_src = await foto_loc.first.get_attribute("src")
            if img_src and img_src.startswith("/"):
                img_src = "https://www.congreso.es" + img_src
                
        # Extraer biografía (CV)
        bio_text = ""
        cv_loc = page.locator(".dip-datos-personales")
        if await cv_loc.count() > 0:
            bio_text = (await cv_loc.first.inner_text()).strip().replace('\n', ' ')
        
        # Partido / Grupo
        partido = ""
        cargo_loc = page.locator(".cargo-diputado")
        if await cargo_loc.count() > 0:
            partido = (await cargo_loc.first.inner_text()).strip()
        
        print(f"[{cod}] Extraído: {nombre} | {partido}")
        await page.close()
        
        return {
            "id": cod,
            "nombre": nombre,
            "foto_url": img_src,
            "partido": partido,
            "biografia": bio_text
        }
    except Exception as e:
        print(f"[{cod}] Vacío, timeout o inactivo")
        await page.close()
        return None

async def run_scraper():
    diputados = []
    print("Iniciando Playwright Async para extraer fotos y biografías oficiales...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        
        # Aceptar cookies en una pestaña inicial
        page = await context.new_page()
        try:
            await page.goto("https://www.congreso.es", timeout=30000, wait_until="load")
            await page.wait_for_timeout(2000)
        except:
            pass
        await page.close()
        
        # Hacer batches de 20 en 20 para no saturar el servidor del Congreso
        batch_size = 20
        total_cods = list(range(1, 380))
        for i in range(0, len(total_cods), batch_size):
            batch = total_cods[i:i+batch_size]
            print(f"Procesando lote de diputados {batch[0]} a {batch[-1]}...")
            tasks = [extract_deputy(context, cod) for cod in batch]
            results = await asyncio.gather(*tasks)
            for res in results:
                if res is not None:
                    diputados.append(res)
                    
        await browser.close()
        
    print(f"\nBúsqueda finalizada. Encontrados {len(diputados)} diputados válidos.")
    if len(diputados) > 200:
        with open(DB_PATH, "w", encoding="utf-8") as f:
            json.dump(diputados, f, ensure_ascii=False, indent=2)
        print("Guardado en diputados_db.json exitosamente.")

if __name__ == "__main__":
    asyncio.run(run_scraper())
