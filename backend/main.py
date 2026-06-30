import os
import json
import asyncio
import re
import time
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import httpx
from bs4 import BeautifulSoup
from google import genai
from google.genai import types
from google.genai.errors import APIError

app = FastAPI(
    title="Figure Price Scanner API",
    description="中古ショップでフィギュアの相場を調べるためのAPI（フェーズ5: 相場チェッカー高速化・再検索）",
    version="0.4.0"
)

# CORS設定（フロントエンドとバックエンドが別ポートで動く場合の保険）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 構造化出力用のPydanticモデル
class FigureAnalysis(BaseModel):
    product_name: str = Field(description="フィギュアの正確な商品名（例: ねんどろいど 初音ミク）")
    series_name: str = Field(description="フィギュアが登場するアニメやゲーム等の作品名（例: キャラクター・ボーカル・シリーズ01 初音ミク）")
    maker_name: str = Field(description="フィギュアを製造したメーカー名（例: グッドスマイルカンパニー）")
    search_keyword: str = Field(description="メルカリやヤフオクなどのフリマアプリで検索する際に最もヒットしやすい最適化された日本語の検索キーワード")
    mercari_price: str = Field(description="AIが知識データベースから推測した、このフィギュアの大体のメルカリ参考相場（カンマなしの数値、例: 5800。どうしても推測できない場合は'確認中'と出力してください）")
    yahoo_price: str = Field(description="AIが知識データベースから推測した、このフィギュアの大体のヤフオク参考相場（カンマなしの数値、例: 5500。どうしても推測できない場合は'確認中'と出力してください）")

# --- インメモリ価格キャッシュの設定 ---
# キャッシュ構造: { keyword: { "mercari_price": "5500", "yahoo_price": "5000", "timestamp": float } }
PRICE_CACHE = {}
CACHE_EXPIRE_SECONDS = 3 * 3600  # キャッシュ有効期限: 3時間

# タイトルから除外するキーワード（まとめ売り、セット、ジャンク等を排除して単体価格を抽出）
EXCLUDE_TITLE_KEYWORDS = ["まとめ", "セット", "set", "引退", "詰め合わせ", "ジャンク", "大量", "アソート", "福袋", "おまとめ"]

# --- スクレイピングヘルパー関数 ---

async def fetch_prices_from_yahoo(keyword: str) -> list[int]:
    """
    ヤフオクの落札履歴（終了品相場）から「まとめ売り」「セット」等を除外した単体の落札価格リストを取得します。
    """
    url = f"https://auctions.yahoo.co.jp/closedsearch/closedsearch?p={keyword}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    prices = []
    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            response = await client.get(url, headers=headers, follow_redirects=True)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                # 商品リストの各要素をパース
                items = soup.select("li.Product, .Product")
                for item in items:
                    title_el = item.select_one(".Product__titleLink, a")
                    price_el = item.select_one(".Product__priceValue, .Price__value, .Product__price")
                    if title_el and price_el:
                        title_text = title_el.get_text().lower()
                        # まとめ売り、セット、ジャンク等を排除
                        if any(kw in title_text for kw in EXCLUDE_TITLE_KEYWORDS):
                            continue
                        
                        price_text = price_el.get_text()
                        num_str = "".join(re.findall(r"\d+", price_text.replace(",", "")))
                        if num_str:
                            price = int(num_str)
                            # 単体フィギュアとして現実的な価格範囲（300円〜60,000円）
                            if 300 < price < 60000:
                                prices.append(price)
                                
                # フォールバック（もしアイテム単位のパースが失敗した場合の旧ロジック互換）
                if not prices:
                    price_elements = soup.find_all(class_=re.compile(r"(Product__priceValue|Price__value|Product__price)"))
                    for elem in price_elements:
                        text = elem.get_text()
                        num_str = "".join(re.findall(r"\d+", text.replace(",", "")))
                        if num_str:
                            price = int(num_str)
                            if 300 < price < 30000:
                                prices.append(price)
    except Exception as e:
        print(f"[Scraper] Yahoo Auction error: {e}")
    return prices

async def fetch_prices_from_mercari(keyword: str) -> list[int]:
    """
    メルカリの検索結果（売り切れ品）から「まとめ売り」「セット」等を除外した単体の販売価格リストを取得します。
    """
    url = f"https://jp.mercari.com/search?keyword={keyword}&status=sold_out"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "ja,en-US;q=0.7,en;q=0.3"
    }
    prices = []
    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            response = await client.get(url, headers=headers, follow_redirects=True)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                
                # 1. JSON-LD（構造化データ）からパース
                json_ld_tags = soup.find_all("script", type="application/ld+json")
                for tag in json_ld_tags:
                    try:
                        data = json.loads(tag.string)
                        if isinstance(data, dict) and data.get("@type") == "ItemList":
                            for item in data.get("itemListElement", []):
                                product = item.get("item", {})
                                title = product.get("name", "").lower()
                                # まとめ売り、セット、ジャンク等を排除
                                if any(kw in title for kw in EXCLUDE_TITLE_KEYWORDS):
                                    continue
                                
                                offers = product.get("offers", {})
                                price = offers.get("price")
                                if price:
                                    val = int(price)
                                    if 300 < val < 60000:
                                        prices.append(val)
                    except Exception:
                        pass
                
                # 2. クラス名によるパース
                if not prices:
                    items = soup.select('li, [class*="Item"], [class*="item"]')
                    for item in items:
                        title_el = item.select_one('[class*="name"], [class*="Name"], a')
                        price_el = item.select_one(class_=re.compile(r"(price|Price)"))
                        if title_el and price_el:
                            title_text = title_el.get_text().lower()
                            if any(kw in title_text for kw in EXCLUDE_TITLE_KEYWORDS):
                                continue
                            price_text = price_el.get_text()
                            num_str = "".join(re.findall(r"\d+", price_text.replace(",", "")))
                            if num_str:
                                price = int(num_str)
                                if 300 < price < 60000:
                                    prices.append(price)
    except Exception as e:
        print(f"[Scraper] Mercari error (likely blocked): {e}")
    return prices

async def fetch_prices_from_surugaya(keyword: str) -> list[int]:
    """
    駿河屋の検索結果から「まとめ売り」「セット」等を除外した単体の中古販売価格リストを取得します。
    """
    url = f"https://www.suruga-ya.jp/search?category=&search_word={keyword}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    prices = []
    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            response = await client.get(url, headers=headers, follow_redirects=True)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                # 各商品コンテナ
                items = soup.select(".item, .product, tr")
                for item in items:
                    title_el = item.select_one(".title, .name, a")
                    price_el = item.select_one(".text-red, .text-price, .price_intax, p.price")
                    if title_el and price_el:
                        title_text = title_el.get_text().lower()
                        if any(kw in title_text for kw in EXCLUDE_TITLE_KEYWORDS):
                            continue
                        
                        price_text = price_el.get_text()
                        num_str = "".join(re.findall(r"\d+", price_text.replace(",", "")))
                        if num_str:
                            price = int(num_str)
                            if 300 < price < 60000:
                                prices.append(price)
                                
                # フォールバック
                if not prices:
                    price_elements = soup.select(".text-red, .text-price, .price_intax, p.price")
                    for elem in price_elements:
                        text = elem.get_text()
                        num_str = "".join(re.findall(r"\d+", text.replace(",", "")))
                        if num_str:
                            price = int(num_str)
                            if 300 < price < 30000:
                                prices.append(price)
    except Exception as e:
        print(f"[Scraper] Surugaya error: {e}")
    return prices

def calculate_average_price(price_list: list[int]) -> str:
    """
    価格リストから外れ値を除外し、500円単位で丸めた平均価格を返却します。
    """
    if not price_list:
        return "確認中"
    
    sorted_prices = sorted(price_list)
    n = len(sorted_prices)
    
    if n >= 4:
        trimmed = sorted_prices[n // 4 : 3 * n // 4]
    else:
        trimmed = sorted_prices
        
    if not trimmed:
        return "確認中"
        
    avg_price = sum(trimmed) // len(trimmed)
    rounded_avg = ((avg_price + 250) // 500) * 500
    
    return str(rounded_avg)

# --- キャッシュ対応価格取得の基底関数 ---

async def get_market_prices(keyword: str) -> tuple[str, str]:
    """
    キーワードを元に、キャッシュまたはスクレイピングからメルカリ・ヤフオクの相場価格を返却します。
    """
    current_time = time.time()
    
    # 1. キャッシュチェック
    if keyword in PRICE_CACHE:
        cache_item = PRICE_CACHE[keyword]
        if current_time - cache_item["timestamp"] < CACHE_EXPIRE_SECONDS:
            print(f"[Cache] Hit for keyword: '{keyword}' (Age: {int(current_time - cache_item['timestamp'])}s)")
            return cache_item["mercari_price"], cache_item["yahoo_price"]
        else:
            print(f"[Cache] Expired for keyword: '{keyword}'")

    # 2. キャッシュにない場合は並行スクレイピング実行
    print(f"[Scanner] Scrape starting for keyword: '{keyword}'")
    yahoo_task = fetch_prices_from_yahoo(keyword)
    mercari_task = fetch_prices_from_mercari(keyword)
    surugaya_task = fetch_prices_from_surugaya(keyword)

    yahoo_prices, mercari_prices, surugaya_prices = await asyncio.gather(
        yahoo_task, mercari_task, surugaya_task
    )

    # 3. 補完・フォールバック
    if not mercari_prices:
        if yahoo_prices:
            mercari_prices = [int(p * 1.1) for p in yahoo_prices]
        elif surugaya_prices:
            mercari_prices = [int(p * 1.05) for p in surugaya_prices]

    if not yahoo_prices:
        if mercari_prices:
            yahoo_prices = [int(p * 0.9) for p in mercari_prices]
        elif surugaya_prices:
            yahoo_prices = [int(p * 0.95) for p in surugaya_prices]

    # 平均値の算出
    final_mercari = calculate_average_price(mercari_prices)
    final_yahoo = calculate_average_price(yahoo_prices)

    # 4. キャッシュに保存
    PRICE_CACHE[keyword] = {
        "mercari_price": final_mercari,
        "yahoo_price": final_yahoo,
        "timestamp": current_time
    }
    
    return final_mercari, final_yahoo

# --- APIエンドポイント ---

@app.post("/api/scan")
async def scan_figure(file: UploadFile = File(...)):
    """
    アップロードされた画像をGemini 2.5 Flashでフィギュアとして識別し、
    その後キャッシュまたはスクレイピングからリアルタイム相場を取得して返却します。
    """
    # APIキーチェック
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="サーバーエラー: 環境変数 'GEMINI_API_KEY' が設定されていません。"
        )

    # 画像データのロード
    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"ファイル読み込みエラー: {str(e)}")

    # Geminiによる画像解析
    try:
        client = genai.Client()
        prompt = (
            "添付された画像に写っているフィギュアを識別し、指示されたスキーマに従って情報を抽出してください。\n"
            "search_keywordは、フリマサイト等で検索した際に最もヒット率が高くなるような、日本語の商品名とシリーズ名を含めた最適な検索キーワード（例: 'ねんどろいど 初音ミク'）にしてください。\n"
            "mercari_priceとyahoo_priceには、あなたの知識データベースを元に、【まとめ売り・セット売り・詰め合わせ・限定豪華版同梱セットなどの高額ケースを除外した、このフィギュア単体（中古・箱あり・良品）】の中古市場における大体の参考相場価格（カンマなしの数値、例: 4500）を推測して出力してください。\n"
            "もしどうしても価格が推測できない場合や不明な場合は'確認中'と出力してください。"
        )

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=file.content_type
                ),
                prompt
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=FigureAnalysis,
                temperature=0.1,
            ),
        )

        analysis = response.parsed
        if not analysis:
            if response.text:
                analysis = FigureAnalysis(**json.loads(response.text))
            else:
                raise Exception("AI応答パースエラー")

    except APIError as e:
        print(f"Gemini API Error: {str(e)}")
        raise HTTPException(status_code=502, detail=f"Gemini API 接続エラー: {str(e)}")
    except Exception as e:
        print(f"AI Analysis Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"画像解析処理エラー: {str(e)}")

    # 初期スキャンは、AIの知識から推測した「大体の相場（なんとなくの相場）」を即時返却して高速化します。
    # ユーザーがより正確なスクレイピング相場を知りたい場合は、フロントの「再検索」をクリックさせます。
    final_mercari_price = analysis.mercari_price
    final_yahoo_price = analysis.yahoo_price

    return {
        "status": "success",
        "data": {
            "product_name": analysis.product_name,
            "series_name": analysis.series_name,
            "maker_name": analysis.maker_name,
            "search_keyword": analysis.search_keyword,
            "mercari_price": final_mercari_price,
            "yahoo_price": final_yahoo_price
        }
    }

@app.get("/api/prices")
async def get_prices_by_keyword(keyword: str):
    """
    キーワードを直接指定して、キャッシュまたはスクレイピングから相場情報を爆速で取得して返却します。
    フロントエンドからの「手動再検索」用軽量エンドポイントです。
    """
    if not keyword or not keyword.strip():
        raise HTTPException(status_code=400, detail="キーワードを指定してください。")
        
    try:
        mercari_price, yahoo_price = await get_market_prices(keyword.strip())
        return {
            "status": "success",
            "data": {
                "search_keyword": keyword.strip(),
                "mercari_price": mercari_price,
                "yahoo_price": yahoo_price
            }
        }
    except Exception as e:
        print(f"Error fetching prices by keyword '{keyword}': {e}")
        raise HTTPException(status_code=500, detail=f"価格情報の再取得中にエラーが発生しました: {str(e)}")

# 静的ファイルの配信用マウント
current_dir = os.path.dirname(os.path.abspath(__file__))
frontend_dir = os.path.join(current_dir, "..", "frontend")

if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
else:
    print(f"Warning: Frontend directory not found at {frontend_dir}.")

if __name__ == "__main__":
    import uvicorn
    # Renderなどのデプロイ環境におけるPORT環境変数に対応
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
