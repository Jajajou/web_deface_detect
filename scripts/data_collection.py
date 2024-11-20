import time
import os
import concurrent.futures
import signal
import sys
import logging
from queue import Queue
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from selenium.common.exceptions import WebDriverException


# Paths to directories for storing screenshots
CLEAN_DIR = 'dataset/clean/'
DEFACED_DIR = 'dataset/defaced/'
PROCESSED_URLS_FILE = 'processed_urls.txt'

# Setup logging
logging.basicConfig(filename='script.log', level=logging.DEBUG, format='%(asctime)s - %(message)s')

# Load URLs from a text file
URLS_FILE = 'dataset/urls.txt'

def load_urls(file_path):
    with open(file_path, 'r') as file:
        urls = [line.strip() for line in file if line.strip()]
    return urls

# Load URLs into a list
URLS = load_urls(URLS_FILE)

# Defacement scripts dictionary
DEFACEMENT_SCRIPTS = {
    'defaced_type_1': "document.body.innerHTML = '<h1>Hacked by XYZ</h1><p>Your website has been defaced!</p>';",
    'defaced_type_2': """
        document.body.style.backgroundColor = 'black'; 
        document.body.style.color = 'yellow';
        document.body.style.filter = 'blur(5px)';
    """,
    'defaced_type_3': """
        var banner = document.createElement('div');
        banner.innerHTML = '<div style="background: red; color: white; text-align: center; padding: 10px;">Hacked by XYZ</div>';
        document.body.insertBefore(banner, document.body.firstChild);
    """,
    'defaced_type_4': """
        var mainContent = document.querySelector('main') || document.body;
        mainContent.innerHTML = '<h1 style="color: red; text-align: center;">Website compromised!</h1><p style="color: black;">This site has been taken over by malicious actors.</p>';
    """
}

# Create directories if they do not exist
os.makedirs(CLEAN_DIR, exist_ok=True)
for deface_type in DEFACEMENT_SCRIPTS.keys():
    os.makedirs(os.path.join(DEFACED_DIR, deface_type), exist_ok=True)

# Signal handler to gracefully handle SIGINT (Ctrl + C)
def signal_handler(sig, frame):
    logging.info("Gracefully shutting down...")
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

# Driver Pool Management
driver_pool = Queue(maxsize=5)

def create_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    try:
        driver = webdriver.Edge(service=Service(EdgeChromiumDriverManager().install()), options=options)
        driver.set_page_load_timeout(30)
    except WebDriverException as e:
        print(f"Error initializing the WebDriver: {e}")
        return None
    
    return driver

def create_driver_pool():
    for _ in range(5):
        driver = create_driver()
        if driver:
            driver_pool.put(driver)
        else:
            print("Failed to create driver. Skipping this instance.")

def get_driver_from_pool():
    try:
        return driver_pool.get(timeout=5)
    except Exception:
        logging.error("Unable to get a WebDriver from pool")
        return create_driver()  # Create a new one if pool is empty

def return_driver_to_pool(driver):
    driver_pool.put(driver)

# Load already processed URLs from a file
def load_processed_urls():
    if not os.path.exists(PROCESSED_URLS_FILE):
        return set()
    with open(PROCESSED_URLS_FILE, 'r') as file:
        return set(line.strip() for line in file)

# Write processed URL to file
def save_processed_url(url):
    with open(PROCESSED_URLS_FILE, 'a') as file:
        file.write(f"{url}\n")

# Capture a clean image
def capture_clean_image(url, index):
    driver = get_driver_from_pool()
    try:
        logging.debug(f"Capturing clean image for URL: {url}")
        driver.get(url)
        WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
        screenshot_path = os.path.join(CLEAN_DIR, f'clean_{index}.png')
        driver.save_screenshot(screenshot_path)
        logging.info(f'Saved clean image: {screenshot_path}')
        save_processed_url(url)
    except TimeoutException:
        logging.error(f"Timeout while trying to load URL: {url}")
    finally:
        return_driver_to_pool(driver)

# Capture a defaced image for each deface type
def capture_defaced_image(url, deface_type, script, index, deface_index):
    driver = get_driver_from_pool()
    try:
        logging.debug(f"Attempting to deface URL: {url} with deface type: {deface_type}")
        driver.get(url)
        WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
        driver.execute_script(script)
        time.sleep(3)  # Give extra time for JavaScript to fully apply
        deface_path = os.path.join(DEFACED_DIR, deface_type, f'defaced_{index}_{deface_index}.png')
        driver.save_screenshot(deface_path)
        logging.info(f'Saved defaced image: {deface_path}')
    except TimeoutException:
        logging.error(f"Timeout while trying to load or deface URL: {url}")
    except Exception as e:
        logging.error(f"Error executing defacement for {url}: {e}")
    finally:
        return_driver_to_pool(driver)

# Capture images for all URLs with all deface types
def capture_images_concurrently():
    deface_types = list(DEFACEMENT_SCRIPTS.keys())
    processed_urls = load_processed_urls()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        try:
            for i, url in enumerate(URLS):
                if url in processed_urls:
                    logging.info(f"Skipping already processed URL: {url}")
                    continue
                futures.append(executor.submit(capture_clean_image, url, i))
                for deface_index, deface_type in enumerate(deface_types):
                    script = DEFACEMENT_SCRIPTS[deface_type]
                    futures.append(executor.submit(capture_defaced_image, url, deface_type, script, i, deface_index))
            concurrent.futures.wait(futures)
        except KeyboardInterrupt:
            logging.info("Graceful shutdown initiated by user interrupt.")
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False)
            sys.exit(0)

# Run the enhanced image capturing
if __name__ == "__main__":
    create_driver_pool()
    capture_images_concurrently()