import time
import random
import uuid
import json
import logging
from pathlib import Path
from urllib.parse import urlparse
from xml.etree import ElementTree as ET
from curl_cffi import requests
from fake_useragent import UserAgent

logging.basicConfig(level=logging.INFO)

SITEMAP_NS = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}


class MTSDownloader:
    def __init__(self, cookie_file="cookies.json", output_dir=None):
        if output_dir is None:
            output_dir = Path(__file__).resolve().parents[2] / "data" / "raw_html"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cookies = self._load_cookies(cookie_file)
        self.ua = UserAgent()
        self.session = requests.Session()
        self.session.headers.update(self._default_headers())
        self.session.cookies.update(self.cookies)

    def _load_cookies(self, cookie_file):
        try:
            with open(cookie_file) as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def _default_headers(self):
        return {
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "ru,en;q=0.9",
            "User-Agent": self.ua.random,
            "x-request-id": str(uuid.uuid4()),
        }

    def _update_headers(self):
        self.session.headers.update({
            "User-Agent": self.ua.random,
            "x-request-id": str(uuid.uuid4()),
        })

    def _wait(self):
        time.sleep(random.uniform(1, 2))

    def download_page(self, url):
        for attempt in range(3):
            self._update_headers()
            try:
                resp = self.session.get(url, timeout=30)
                if resp.status_code == 200:
                    return resp.text
                logging.warning(f"Код {resp.status_code}, попытка {attempt+1}/3")
            except Exception as e:
                logging.error(e)
            self._wait()
        raise Exception(f"Не удалось скачать {url}")

    def save(self, html, name, subsection):
        subdir = self.output_dir / subsection
        subdir.mkdir(parents=True, exist_ok=True)
        path = subdir / f"{name}.html"
        path.write_text(html, encoding='utf-8')
        logging.info(f"Сохранено {path}")

    def close(self):
        self.session.close()

    def get_urls(self):
        sitemap_url = "https://support.mts.ru/sitemap.xml"
        logging.info(f"Загружаю {sitemap_url}")
        resp = self.session.get(sitemap_url, timeout=30)
        root = ET.fromstring(resp.content)
        urls = [loc.text for loc in root.findall('.//ns:loc', SITEMAP_NS)]
        logging.info(f"Найдено {len(urls)} страниц")
        return urls

    @staticmethod
    def parse_url(url):
        parts = urlparse(url).path.strip('/').split('/')
        if not parts or parts == ['']:
            return 'root', 'index'
        subsection = parts[0]
        tail = parts[1:] if len(parts) > 1 else ['index']
        name = '__'.join(tail)
        return subsection, name


downloader = MTSDownloader()
try:
    urls = downloader.get_urls()

    for u in urls[:5]:
        logging.info("PREVIEW %s -> %s", u, downloader.parse_url(u))

    for url in urls:
        subsection, name = downloader.parse_url(url)

        # ===== ДОБАВЛЕНО: проверка существования файла =====
        file_path = downloader.output_dir / subsection / f"{name}.html"
        if file_path.exists():
            logging.info(f"Файл уже существует, пропускаем {url}")
            continue

        # ===== ДОБАВЛЕНО: обработка ошибок с записью в fails.txt =====
        try:
            html = downloader.download_page(url)
            downloader.save(html, name, subsection)
        except Exception as e:
            logging.error(f"Не удалось скачать {url}: {e}")
            with open("fails.txt", "a", encoding="utf-8") as f:
                f.write(url + "\n")

        downloader._wait()
finally:
    downloader.close()